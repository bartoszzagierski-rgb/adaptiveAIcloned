"""Handel na zywo przez MT5 (konto demo lub realne) wytrenowanym agentem.

Zabezpieczenia:
  * tryb 'demo' odmowi pracy na koncie realnym, tryb 'live' - na koncie demo,
  * 'live' wymaga flagi --i-understand-live-risk i wpisania numeru konta,
  * --dry-run: zadnych zlecen, pozycje tylko na papierze (dobry pierwszy krok),
  * limit dziennej straty (zamyka pozycje bota i zatrzymuje), limit spreadu przy otwieraniu,
  * max pozycja z konfiguracji modelu, bot dotyka tylko swoich pozycji (magic number),
  * plik STOP w katalogu roboczym zatrzymuje bota, Ctrl+C - bezpieczne wyjscie,
  * opcjonalny nadzor LLM (LLMSupervisor) - wstrzymuje akcje, ktore uzna za ryzykowne,
  * kazda decyzja zapisywana do CSV.
"""
import csv
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np

from .features import make_obs
from .mt5_data import bars_to_gym, ticks_to_gym, timeframe_const

log = logging.getLogger(__name__)

DEFAULT_MAGIC = 770077


class LiveError(RuntimeError):
    pass


def _utc(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc)


class TickFeed:
    """Bufor ostatnich obs_len tickow; decyzja co step_len nowych tickow."""

    def __init__(self, mt5, cfg):
        self.mt5, self.cfg = mt5, cfg
        self.buf = deque(maxlen=cfg.obs_len)
        self.last_msc = 0
        self.new_since_decision = 0

    def _append(self, ticks):
        if ticks is None or len(ticks) == 0:
            return 0
        ticks = ticks[ticks['time_msc'] > self.last_msc]
        if len(ticks) == 0:
            return 0
        df = ticks_to_gym(ticks, self.cfg.price_source)
        for row in df[self.cfg.features].to_numpy():
            self.buf.append(row)
        self.last_msc = int(ticks['time_msc'].max())
        return len(df)

    def warmup(self):
        tick = self.mt5.symbol_info_tick(self.cfg.symbol)
        if tick is None:
            raise LiveError('Brak ceny dla {}: {}'.format(self.cfg.symbol, self.mt5.last_error()))
        for hours in (1, 6, 24, 72):
            self.buf.clear()
            self.last_msc = 0
            self._append(self.mt5.copy_ticks_range(self.cfg.symbol, _utc(tick.time - hours * 3600),
                                                   _utc(tick.time + 60), self.mt5.COPY_TICKS_ALL))
            if len(self.buf) == self.cfg.obs_len:
                return
        raise LiveError('Za malo historii tickow do rozgrzania bufora ({} < {})'.format(len(self.buf), self.cfg.obs_len))

    def poll(self):
        start = _utc(self.last_msc / 1000.0)
        self.new_since_decision += self._append(
            self.mt5.copy_ticks_from(self.cfg.symbol, start, 100_000, self.mt5.COPY_TICKS_ALL))
        if self.new_since_decision >= self.cfg.step_len and len(self.buf) == self.cfg.obs_len:
            self.new_since_decision = 0
            return True
        return False

    def window(self):
        return np.array(self.buf, dtype=np.float64)


class BarFeed:
    """Ostatnie obs_len zamknietych swiec; decyzja co step_len nowych swiec."""

    def __init__(self, mt5, cfg):
        self.mt5, self.cfg = mt5, cfg
        self.tf = timeframe_const(cfg.timeframe)
        self.point = mt5.symbol_info(cfg.symbol).point
        self.last_time = None
        self.new_since_decision = 0
        self._window = None

    def _load(self):
        rates = self.mt5.copy_rates_from_pos(self.cfg.symbol, self.tf, 1, self.cfg.obs_len)  # 1 = ostatnia zamknieta
        if rates is None or len(rates) < self.cfg.obs_len:
            raise LiveError('Za malo swiec: {}'.format(self.mt5.last_error()))
        return rates

    def warmup(self):
        rates = self._load()
        self._window = bars_to_gym(rates, self.point)[self.cfg.features].to_numpy()
        self.last_time = int(rates['time'][-1])

    def poll(self):
        rates = self._load()
        t = int(rates['time'][-1])
        if t == self.last_time:
            return False
        self.new_since_decision += int(np.sum(rates['time'] > self.last_time))
        self.last_time = t
        self._window = bars_to_gym(rates, self.point)[self.cfg.features].to_numpy()
        if self.new_since_decision >= self.cfg.step_len:
            self.new_since_decision = 0
            return True
        return False

    def window(self):
        return np.asarray(self._window, dtype=np.float64)


class PaperBook:
    """Pozycje 'na papierze' dla --dry-run (ten sam ksztalt co mt5.positions_get)."""

    def __init__(self):
        self.positions = []
        self._ticket = 0
        self.realized = 0.0

    def send(self, mt5, request, price):
        if request.get('position'):
            p = next(p for p in self.positions if p.ticket == request['position'])
            sign = 1 if p.type == mt5.POSITION_TYPE_BUY else -1
            self.realized += (price - p.price_open) * sign * request['volume']
            p.volume = round(p.volume - request['volume'], 8)
            if p.volume <= 1e-9:
                self.positions.remove(p)
        else:
            self._ticket += 1
            self.positions.append(SimpleNamespace(
                ticket=self._ticket, symbol=request['symbol'], volume=request['volume'], price_open=price,
                type=mt5.POSITION_TYPE_BUY if request['type'] == mt5.ORDER_TYPE_BUY else mt5.POSITION_TYPE_SELL,
                magic=request['magic'], time=int(time.time())))
        return SimpleNamespace(retcode=mt5.TRADE_RETCODE_DONE, comment='paper', price=price, volume=request['volume'])


class LiveTrader:
    def __init__(self, cfg, policy, mt5, mode='demo', dry_run=False, lot=0.01, max_daily_loss=None,
                 max_spread=None, supervisor=None, poll_interval=1.0, magic=DEFAULT_MAGIC, close_on_exit=False,
                 stop_file='STOP', log_path='live_trades.csv', confirm_live=False, deviation=20):
        if mode not in ('demo', 'live'):
            raise ValueError('mode musi byc demo albo live')
        self.cfg, self.policy, self.mt5 = cfg, policy, mt5
        self.mode, self.dry_run, self.lot = mode, dry_run, lot
        self.max_daily_loss = max_daily_loss
        self.max_spread = max_spread if max_spread is not None else cfg.max_spread
        self.supervisor, self.poll_interval, self.magic = supervisor, poll_interval, magic
        self.close_on_exit, self.stop_file, self.log_path = close_on_exit, stop_file, log_path
        self.confirm_live, self.deviation = confirm_live, deviation
        self.paper = PaperBook() if dry_run else None
        self.running = False
        self.decisions = 0

    # ---------------- start / kontrola konta ----------------
    def start(self):
        mt5 = self.mt5
        acc = mt5.account_info()
        if acc is None:
            raise LiveError('Brak informacji o koncie: {}'.format(mt5.last_error()))
        is_real = acc.trade_mode == mt5.ACCOUNT_TRADE_MODE_REAL
        if self.mode == 'demo' and is_real:
            raise LiveError('Tryb demo, ale zalogowane konto {} jest REALNE. Zaloguj sie na konto demo.'.format(acc.login))
        if self.mode == 'live':
            if not is_real:
                raise LiveError('Tryb live wymaga konta realnego (zalogowane {} to nie konto realne).'.format(acc.login))
            if not self.confirm_live:
                raise LiveError('Handel live nie zostal potwierdzony.')
            if self.max_daily_loss is None:
                raise LiveError('Handel live wymaga ustawienia --max-daily-loss.')
        if not self.dry_run:
            term = mt5.terminal_info()
            if not (acc.trade_allowed and term.trade_allowed):
                raise LiveError('Handel algorytmiczny wylaczony - wlacz przycisk "Algo Trading" w terminalu MT5.')
        if not mt5.symbol_select(self.cfg.symbol, True):
            raise LiveError('Symbol {} niedostepny'.format(self.cfg.symbol))
        self.sym = mt5.symbol_info(self.cfg.symbol)
        self.filling = (mt5.ORDER_FILLING_FOK if self.sym.filling_mode & 1 else
                        mt5.ORDER_FILLING_IOC if self.sym.filling_mode & 2 else mt5.ORDER_FILLING_RETURN)
        self.volume = self._norm_volume(self.lot)
        self.feed = TickFeed(mt5, self.cfg) if self.cfg.source == 'ticks' else BarFeed(mt5, self.cfg)
        self.feed.warmup()
        self.day = None
        self._roll_day(acc)
        self.running = True
        log.info('Start: %s%s, konto %s (%s), %s, lot %.2f na jednostke, max pozycja %d, max spread %s, sesja %s-%s',
                 self.mode.upper(), ' DRY-RUN' if self.dry_run else '', acc.login, acc.server,
                 self.cfg.symbol, self.volume, self.cfg.max_position, self.max_spread,
                 self.cfg.session_start or '-', self.cfg.flat_after or '-')

    def _norm_volume(self, v):
        step = self.sym.volume_step
        v = round(round(v / step) * step, 8)
        return float(min(max(v, self.sym.volume_min), self.sym.volume_max))

    def _roll_day(self, acc):
        today = datetime.now(timezone.utc).date()
        if today != self.day:
            self.day = today
            self.day_start_equity = acc.equity
            self.day_start_paper = self.paper.realized if self.paper else 0.0

    # ---------------- pozycje ----------------
    def _positions(self):
        if self.paper:
            return list(self.paper.positions)
        return [p for p in (self.mt5.positions_get(symbol=self.cfg.symbol) or ()) if p.magic == self.magic]

    def position_state(self, price):
        """(pozycja w jednostkach, niezrealizowany wynik w jednostkach ceny na jednostke) - jak w TradingGym."""
        units, unreal = 0.0, 0.0
        for p in self._positions():
            sign = 1 if p.type == self.mt5.POSITION_TYPE_BUY else -1
            u = p.volume / self.volume
            units += sign * u
            unreal += (price - p.price_open) * sign * u
        return units, unreal - abs(units) * self.cfg.fee

    # ---------------- zlecenia ----------------
    def _send(self, order_type, volume, position=None, comment='tradebot'):
        mt5 = self.mt5
        tick = mt5.symbol_info_tick(self.cfg.symbol)
        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        req = {'action': mt5.TRADE_ACTION_DEAL, 'symbol': self.cfg.symbol, 'volume': volume, 'type': order_type,
               'price': price, 'deviation': self.deviation, 'magic': self.magic, 'comment': comment,
               'type_time': mt5.ORDER_TIME_GTC, 'type_filling': self.filling}
        if position is not None:
            req['position'] = position
        res = self.paper.send(mt5, req, price) if self.paper else mt5.order_send(req)
        if res is None:
            raise LiveError('order_send zwrocil None: {}'.format(mt5.last_error()))
        ok = res.retcode == mt5.TRADE_RETCODE_DONE
        (log.info if ok else log.error)('%s %s %.2f @ %.5f -> %s %s', 'BUY' if order_type == mt5.ORDER_TYPE_BUY
                                        else 'SELL', self.cfg.symbol, volume, price, res.retcode, res.comment)
        return ok, res

    def _close_one(self, pos_type):
        """Zamyka 1 jednostke z najstarszej pozycji danego typu."""
        mt5 = self.mt5
        candidates = sorted((p for p in self._positions() if p.type == pos_type), key=lambda p: (p.time, p.ticket))
        if not candidates:
            return False, None
        p = candidates[0]
        order = mt5.ORDER_TYPE_SELL if pos_type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        return self._send(order, min(self.volume, p.volume), position=p.ticket, comment='tradebot close')

    def close_all(self):
        mt5 = self.mt5
        for p in self._positions():
            order = mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            self._send(order, p.volume, position=p.ticket, comment='tradebot close all')

    def execute(self, action, units, spread):
        """Ta sama logika co TradingGym: kup = zamknij short albo dodaj long; sprzedaj odwrotnie."""
        mt5, maxp = self.mt5, self.cfg.max_position
        opening = (action == 1 and units >= -1e-9) or (action == 2 and units <= 1e-9)
        if opening and self.max_spread is not None and spread > self.max_spread:
            return 'skip: spread {:.6f} > {:.6f}'.format(spread, self.max_spread)
        if action == 1:
            if units < -1e-9:
                return 'close short: {}'.format(self._close_one(mt5.POSITION_TYPE_SELL)[0])
            if units < maxp - 1e-9:
                return 'open long: {}'.format(self._send(mt5.ORDER_TYPE_BUY, self.volume)[0])
            return 'skip: max long'
        if action == 2:
            if units > 1e-9:
                return 'close long: {}'.format(self._close_one(mt5.POSITION_TYPE_BUY)[0])
            if units > -maxp + 1e-9:
                return 'open short: {}'.format(self._send(mt5.ORDER_TYPE_SELL, self.volume)[0])
            return 'skip: max short'
        return 'hold'

    # ---------------- petla ----------------
    def check_risk(self):
        if self.stop_file and os.path.exists(self.stop_file):
            log.warning('Znaleziono plik %s - zatrzymuje.', self.stop_file)
            return False
        acc = self.mt5.account_info()
        if acc is None:
            log.error('Utracono polaczenie z MT5: %s', self.mt5.last_error())
            return False
        self._roll_day(acc)
        if self.max_daily_loss is not None and not self.dry_run:
            loss = self.day_start_equity - acc.equity
            if loss >= self.max_daily_loss:
                log.error('Dzienna strata %.2f >= limit %.2f - zamykam pozycje bota i koncze.', loss, self.max_daily_loss)
                self.close_all()
                return False
        return True

    def iteration(self):
        """Jedna iteracja petli. Zwraca dict z decyzja albo None (brak nowej decyzji)."""
        if not self.check_risk():
            self.running = False
            return None
        ready = self.feed.poll()
        if not self.in_session():
            return None
        if not ready:
            return None
        window = self.feed.window()
        price = float(window[-1, 0])
        units, unreal = self.position_state(price)
        obs = make_obs(window, units, self.cfg.max_position, unreal)
        action = self.policy.act(obs, window, units, unreal)
        approved, why = True, ''
        if action != 0 and self.supervisor is not None:
            approved, why = self.supervisor.review(action, window, units, unreal)
        tick = self.mt5.symbol_info_tick(self.cfg.symbol)
        spread = tick.ask - tick.bid
        result = self.execute(action, units, spread) if approved else 'vetoed by LLM'
        self.decisions += 1
        rec = {'time': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'mode': self.mode,
               'dry_run': self.dry_run, 'symbol': self.cfg.symbol, 'price': price, 'spread': spread,
               'position_units': units, 'unrealized': round(unreal, 6), 'action': {0: 'hold', 1: 'buy', 2: 'sell'}[action],
               'result': result, 'llm_reason': why}
        self._log(rec)
        if action != 0:
            log.info('Decyzja: %s (pozycja %.2f) -> %s %s', rec['action'], units, result, why)
        return rec

    def server_time(self):
        """Godzina 'HH:MM' czasu serwera MT5 (czasy tickow w MT5 sa w strefie serwera)."""
        tick = self.mt5.symbol_info_tick(self.cfg.symbol)
        return datetime.fromtimestamp(tick.time, tz=timezone.utc).strftime('%H:%M')

    def in_session(self):
        """Odwzorowuje dzien z treningu: przed session_start nie handluje, po flat_after zamyka pozycje."""
        if not (self.cfg.session_start or self.cfg.flat_after):
            return True
        hm = self.server_time()
        if self.cfg.flat_after and hm >= self.cfg.flat_after:
            if self._positions():
                log.info('Koniec sesji (%s >= %s czasu serwera) - zamykam pozycje bota.', hm, self.cfg.flat_after)
                self.close_all()
            return False
        return not (self.cfg.session_start and hm < self.cfg.session_start)

    def _log(self, rec):
        if not self.log_path:
            return
        new = not os.path.exists(self.log_path)
        with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rec))
            if new:
                w.writeheader()
            w.writerow(rec)

    def run(self, max_iterations=None, sleep=time.sleep):
        self.start()
        i = 0
        try:
            while self.running and (max_iterations is None or i < max_iterations):
                self.iteration()
                i += 1
                sleep(self.poll_interval)
        except KeyboardInterrupt:
            log.info('Przerwano (Ctrl+C).')
        finally:
            if self.close_on_exit:
                log.info('Zamykam pozycje bota (--close-on-exit).')
                self.close_all()
            self.running = False
