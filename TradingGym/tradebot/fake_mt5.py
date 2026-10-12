"""Symulator API MetaTrader5 (ten sam interfejs co pakiet MetaTrader5).

Pozwala przetestowac caly pipeline (pobieranie danych, trening, handel demo) bez MT5,
np. na Linuksie albo zanim skonfigurujesz terminal. Uzycie: --fake-mt5 w CLI.
Ceny to losowy spacer - wyniki nie maja zadnego znaczenia rynkowego.
"""
import time as _time
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np

ACCOUNT_TRADE_MODE_DEMO, ACCOUNT_TRADE_MODE_CONTEST, ACCOUNT_TRADE_MODE_REAL = 0, 1, 2
ORDER_TYPE_BUY, ORDER_TYPE_SELL = 0, 1
POSITION_TYPE_BUY, POSITION_TYPE_SELL = 0, 1
TRADE_ACTION_DEAL = 1
ORDER_TIME_GTC = 0
ORDER_FILLING_FOK, ORDER_FILLING_IOC, ORDER_FILLING_RETURN = 0, 1, 2
TRADE_RETCODE_DONE = 10009
COPY_TICKS_ALL = -1
TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_M15, TIMEFRAME_M30 = 1, 5, 15, 30
TIMEFRAME_H1, TIMEFRAME_H4, TIMEFRAME_D1 = 16385, 16388, 16408
_TF_SECONDS = {1: 60, 5: 300, 15: 900, 30: 1800, 16385: 3600, 16388: 14400, 16408: 86400}

TICK_DTYPE = np.dtype([('time', '<i8'), ('bid', '<f8'), ('ask', '<f8'), ('last', '<f8'), ('volume', '<u8'),
                       ('time_msc', '<i8'), ('flags', '<u4'), ('volume_real', '<f8')])
RATE_DTYPE = np.dtype([('time', '<i8'), ('open', '<f8'), ('high', '<f8'), ('low', '<f8'), ('close', '<f8'),
                       ('tick_volume', '<u8'), ('spread', '<i4'), ('real_volume', '<u8')])


def _ts(d):
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp()


class FakeMT5:
    """tick_ms: odstep miedzy tickami. now: 'aktualny' czas symulacji (sekundy UTC)."""

    def __init__(self, trade_mode=ACCOUNT_TRADE_MODE_DEMO, base_price=1.1000, tick_ms=500, seed=0,
                 now=None, realtime=False):
        for k, v in globals().items():
            if k.isupper():
                setattr(self, k, v)
        self.trade_mode = trade_mode
        self.base = base_price
        self.tick_ms = tick_ms
        self.seed = seed
        self.realtime = realtime
        self._now = now if now is not None else _time.time()
        self._t0 = _time.time()
        self.positions = []
        self.orders = []
        self._ticket = 1000
        self.balance = 10000.0
        self.point = 0.00001

    # --- czas / ceny ---
    def now(self):
        return self._now + (_time.time() - self._t0 if self.realtime else 0)

    def advance(self, seconds):
        self._now += seconds

    def _prices(self, msc):
        """Deterministyczny losowy spacer: cena zalezy tylko od numeru ticka."""
        idx = msc // self.tick_ms
        block, pos = idx // 4096, idx % 4096
        out = np.empty(len(idx))
        for b in np.unique(block):
            sel = block == b
            steps = np.random.default_rng((self.seed, int(b))).normal(0, 2e-5, 4096).cumsum()
            offset = np.random.default_rng((self.seed, int(b), 7)).normal(0, 1e-4)
            out[sel] = self.base * (1 + 0.002 * np.sin(idx[sel] / 50000.0)) + steps[pos[sel]] + offset
        return out

    def _ticks(self, start_s, end_s):
        first = int(np.ceil(start_s * 1000 / self.tick_ms))
        last = int(min(end_s, self.now()) * 1000 // self.tick_ms)
        if last < first:
            return np.zeros(0, dtype=TICK_DTYPE)
        msc = np.arange(first, last + 1, dtype=np.int64) * self.tick_ms
        mid = self._prices(msc)
        spread = 10 * self.point
        arr = np.zeros(len(msc), dtype=TICK_DTYPE)
        arr['time_msc'] = msc
        arr['time'] = msc // 1000
        arr['bid'] = np.round(mid - spread / 2, 5)
        arr['ask'] = np.round(mid + spread / 2, 5)
        arr['flags'] = 6
        return arr

    # --- API zgodne z MetaTrader5 ---
    def initialize(self, **kwargs):
        return True

    def shutdown(self):
        return True

    def last_error(self):
        return (1, 'Success')

    def account_info(self):
        equity = self.balance + sum(self._profit(p) for p in self.positions)
        return SimpleNamespace(login=12345678, server='Fake-Demo', trade_mode=self.trade_mode, balance=self.balance,
                               equity=equity, currency='USD', trade_allowed=True, leverage=100)

    def terminal_info(self):
        return SimpleNamespace(trade_allowed=True, connected=True)

    def symbol_select(self, symbol, enable=True):
        return True

    def symbol_info(self, symbol):
        return SimpleNamespace(name=symbol, point=self.point, digits=5, volume_min=0.01, volume_max=100.0,
                               volume_step=0.01, trade_contract_size=100000.0, filling_mode=1, visible=True)

    def symbol_info_tick(self, symbol):
        t = self._ticks(self.now() - 5, self.now())[-1]
        return SimpleNamespace(time=int(t['time']), time_msc=int(t['time_msc']), bid=float(t['bid']),
                               ask=float(t['ask']), last=0.0, volume=0)

    def copy_ticks_range(self, symbol, date_from, date_to, flags):
        return self._ticks(_ts(date_from), _ts(date_to))

    def copy_ticks_from(self, symbol, date_from, count, flags):
        return self._ticks(_ts(date_from), self.now())[:count]

    def _rates(self, start_s, end_s, timeframe):
        sec = _TF_SECONDS[timeframe]
        first = int(np.ceil(start_s / sec))
        last = int(min(end_s, self.now()) // sec)
        rows = []
        for b in range(first, last + 1):
            t = self._ticks(b * sec, b * sec + sec - 0.001)
            if not len(t):
                continue
            mid = (t['bid'] + t['ask']) / 2
            rows.append((b * sec, mid[0], mid.max(), mid.min(), mid[-1], len(t), 10, 0))
        return np.array(rows, dtype=RATE_DTYPE)

    def copy_rates_range(self, symbol, timeframe, date_from, date_to):
        return self._rates(_ts(date_from), _ts(date_to), timeframe)

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        sec = _TF_SECONDS[timeframe]
        end = (self.now() // sec - start_pos) * sec
        return self._rates(end - (count - 1) * sec, end, timeframe)

    def positions_get(self, symbol=None):
        return tuple(p for p in self.positions if symbol is None or p.symbol == symbol)

    def _profit(self, p):
        tick = self.symbol_info_tick(p.symbol)
        price = tick.bid if p.type == POSITION_TYPE_BUY else tick.ask
        sign = 1 if p.type == POSITION_TYPE_BUY else -1
        return (price - p.price_open) * sign * p.volume * 100000.0

    def order_send(self, request):
        self.orders.append(dict(request))
        tick = self.symbol_info_tick(request['symbol'])
        price = tick.ask if request['type'] == ORDER_TYPE_BUY else tick.bid
        vol = request['volume']
        if request.get('position'):
            p = next((p for p in self.positions if p.ticket == request['position']), None)
            if p is None:
                return SimpleNamespace(retcode=10013, comment='Invalid position', order=0, price=price, volume=0)
            closed = SimpleNamespace(**vars(p))
            closed.volume = vol
            self.balance += self._profit(closed)
            p.volume = round(p.volume - vol, 2)
            if p.volume <= 1e-9:
                self.positions.remove(p)
        else:
            self._ticket += 1
            self.positions.append(SimpleNamespace(
                ticket=self._ticket, symbol=request['symbol'], volume=vol, price_open=price,
                type=POSITION_TYPE_BUY if request['type'] == ORDER_TYPE_BUY else POSITION_TYPE_SELL,
                magic=request.get('magic', 0), time=int(self.now()), comment=request.get('comment', '')))
        self._ticket += 1
        return SimpleNamespace(retcode=TRADE_RETCODE_DONE, comment='Request executed', order=self._ticket,
                               price=price, volume=vol)


def default_fake():
    """Symulator z 'teraz' = poniedzialek 12:00 UTC (stabilne dane do testow i demo)."""
    return FakeMT5(now=datetime(2024, 3, 4, 12, 0, tzinfo=timezone.utc).timestamp())
