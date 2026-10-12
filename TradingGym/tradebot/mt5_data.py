"""Polaczenie z MetaTrader 5 i pobieranie danych do formatu TradingGym.

Wymaga: Windows, zainstalowany i zalogowany terminal MT5, `pip install MetaTrader5`.
Dane logowania (opcjonalne, jesli terminal jest juz zalogowany) czytane sa ze zmiennych
srodowiskowych MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH - nie zapisuj ich w plikach.
"""
import logging
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_MT5 = None  # modul MetaTrader5 albo FakeMT5 (testy / --fake-mt5)


def use_module(module):
    """Podmienia modul MT5 (np. na tradebot.fake_mt5.FakeMT5())."""
    global _MT5
    _MT5 = module


def get_mt5():
    global _MT5
    if _MT5 is None:
        try:
            import MetaTrader5
        except ImportError as e:
            raise RuntimeError('Brak pakietu MetaTrader5. Zainstaluj na Windows: python -m pip install MetaTrader5 '
                               '(albo uzyj --fake-mt5 do testow bez MT5).') from e
        _MT5 = MetaTrader5
    return _MT5


def connect():
    mt5 = get_mt5()
    kwargs = {}
    if os.environ.get('MT5_PATH'):
        kwargs['path'] = os.environ['MT5_PATH']
    if os.environ.get('MT5_LOGIN'):
        kwargs.update(login=int(os.environ['MT5_LOGIN']),
                      password=os.environ.get('MT5_PASSWORD', ''),
                      server=os.environ.get('MT5_SERVER', ''))
    if not mt5.initialize(**kwargs):
        raise RuntimeError('Nie udalo sie polaczyc z MT5: {}. Czy terminal jest uruchomiony i zalogowany?'
                           .format(mt5.last_error()))
    info = mt5.account_info()
    log.info('Polaczono z MT5: konto %s, serwer %s, tryb %s', info.login, info.server,
             {0: 'DEMO', 1: 'CONTEST', 2: 'REAL'}.get(info.trade_mode, info.trade_mode))
    return mt5


def resolve_symbol(symbol):
    """Zwraca nazwe symbolu u brokera, np. XAUUSD -> XAUUSDm / XAUUSD.pro / GOLD."""
    mt5 = get_mt5()
    if mt5.symbol_select(symbol, True) and mt5.symbol_info(symbol) is not None:
        return symbol
    aliases = {'XAUUSD': ['GOLD'], 'XAGUSD': ['SILVER']}.get(symbol.upper(), [])
    candidates = []
    for pattern in [symbol] + aliases:
        for s in mt5.symbols_get(group='*{}*'.format(pattern)) or ():
            candidates.append(s.name)
    candidates = sorted(set(candidates), key=len)
    for name in candidates:
        if mt5.symbol_select(name, True):
            log.info('Symbol %s u tego brokera nazywa sie %s', symbol, name)
            return name
    raise RuntimeError('Symbol {} niedostepny w MT5 (podobne: {}). Podaj dokladna nazwe z okna "Obserwacja rynku".'
                       .format(symbol, ', '.join(candidates) or 'brak'))


def _select(mt5, symbol):
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError('Symbol {} niedostepny w MT5: {}'.format(symbol, mt5.last_error()))
    return mt5.symbol_info(symbol)


def _utc(d):
    if isinstance(d, str):
        d = datetime.fromisoformat(d)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def fetch_ticks(symbol, start, end, chunk_days=1):
    """Pobiera ticki dzien po dniu (MT5 ma limity na jedno zapytanie)."""
    mt5 = get_mt5()
    _select(mt5, symbol)
    start, end = _utc(start), _utc(end)
    parts = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days), end)
        ticks = mt5.copy_ticks_range(symbol, cur, nxt, mt5.COPY_TICKS_ALL)
        if ticks is None:
            log.warning('Brak tickow %s..%s: %s', cur, nxt, mt5.last_error())
        elif len(ticks):
            parts.append(pd.DataFrame(ticks))
            log.info('%s: %d tickow %s', symbol, len(ticks), cur.date())
        cur = nxt
    if not parts:
        raise RuntimeError('MT5 nie zwrocil zadnych tickow dla {} w zakresie {} - {}'.format(symbol, start, end))
    raw = pd.concat(parts, ignore_index=True)
    return raw.drop_duplicates(subset=[c for c in ('time_msc', 'bid', 'ask', 'last') if c in raw.columns])


def fetch_bars(symbol, timeframe, start, end, chunk_days=90):
    """Swiece w kawalkach po chunk_days. Uwaga: MT5 zwroci najwyzej tyle swiec, ile ustawiono w
    Narzedzia > Opcje > Wykresy > "Maks. swiec na wykresie" - dla 2 lat M5 ustaw "Unlimited"."""
    mt5 = get_mt5()
    info = _select(mt5, symbol)
    tf = timeframe_const(timeframe)
    start, end = _utc(start), _utc(end)
    parts, cur = [], start
    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days), end)
        rates = mt5.copy_rates_range(symbol, tf, cur, nxt)
        if rates is not None and len(rates):
            parts.append(pd.DataFrame(rates))
        cur = nxt
    if not parts:
        raise RuntimeError('MT5 nie zwrocil swiec dla {}: {}'.format(symbol, mt5.last_error()))
    raw = pd.concat(parts, ignore_index=True).drop_duplicates(subset='time')
    first = pd.to_datetime(raw['time'].min(), unit='s')
    if first > pd.Timestamp(start.replace(tzinfo=None)) + pd.Timedelta(days=7):
        log.warning('Historia zaczyna sie dopiero %s (chciano %s). Zwieksz "Maks. swiec na wykresie" w MT5 '
                    'na Unlimited i pobierz ponownie.', first, start.date())
    return raw, info.point


def timeframe_const(name):
    mt5 = get_mt5()
    try:
        return getattr(mt5, 'TIMEFRAME_' + name.upper())
    except AttributeError:
        raise ValueError('Nieznany interwal {} (np. M1, M5, M15, H1, H4, D1)'.format(name))


# ---- konwersja do formatu TradingGym ----------------------------------------------------

def ticks_to_gym(raw, price_source='mid'):
    df = pd.DataFrame(raw)
    if df.empty:
        raise ValueError('Puste dane tickowe')
    t = pd.to_datetime(df['time_msc'], unit='ms') if 'time_msc' in df else pd.to_datetime(df['time'], unit='s')
    bid = df['bid'].astype(float)
    ask = df['ask'].astype(float)
    price = df['last'].astype(float) if price_source == 'last' else (bid + ask) / 2
    if 'volume_real' in df and (df['volume_real'] > 0).any():
        vol = df['volume_real'].astype(float)
    elif 'volume' in df:
        vol = df['volume'].astype(float)
    else:
        vol = pd.Series(0.0, index=df.index)
    out = pd.DataFrame({'datetime': t, 'price': price, 'bid': bid, 'ask': ask,
                        'spread': ask - bid, 'volume': vol})
    out = out[(bid > 0) & (ask > 0) & (price > 0)]
    return _finalize(out)


def bars_to_gym(raw, point):
    df = pd.DataFrame(raw)
    if df.empty:
        raise ValueError('Puste dane swiecowe')
    close = df['close'].astype(float)
    spread = df['spread'].astype(float) * point if 'spread' in df else pd.Series(0.0, index=df.index)
    vol = df['real_volume'] if 'real_volume' in df and (df['real_volume'] > 0).any() else df['tick_volume']
    out = pd.DataFrame({'datetime': pd.to_datetime(df['time'], unit='s'), 'price': close,
                        'open': df['open'].astype(float), 'high': df['high'].astype(float),
                        'low': df['low'].astype(float), 'range': (df['high'] - df['low']).astype(float),
                        'bid': close, 'ask': close + spread,
                        'spread': spread, 'volume': vol.astype(float)})
    return _finalize(out)


def _finalize(out):
    """Sortuje, ustawia indeks 0..N i serial_number (0 = pierwszy rekord dnia) - wymagania TradingGym."""
    out = out.sort_values('datetime', kind='stable').reset_index(drop=True)
    out['serial_number'] = out.groupby(out['datetime'].dt.date).cumcount()
    return out


def drop_short_days(df, min_len):
    """Usuwa dni krotsze niz min_len rekordow (TradingGym nie obsluguje dnia krotszego niz okno)."""
    day = df['datetime'].dt.date
    sizes = day.map(day.value_counts())
    kept = df[sizes >= min_len]
    dropped = day.nunique() - kept['datetime'].dt.date.nunique()
    if dropped:
        log.info('Pominieto %d dni z mniej niz %d rekordami', dropped, min_len)
    if kept.empty:
        raise ValueError('Zaden dzien nie ma >= {} rekordow. Zmniejsz obs_len/step_len albo pobierz wiecej danych.'
                         .format(min_len))
    return _finalize(kept.drop(columns='serial_number'))


def split_days(df, test_frac=0.2):
    """Podzial chronologiczny po dniach: poczatek -> trening, koniec -> test."""
    days = sorted(df['datetime'].dt.date.unique())
    if len(days) < 2 or test_frac <= 0:
        return df, df
    n_test = max(1, int(round(len(days) * test_frac)))
    cut = days[-n_test]
    d = df['datetime'].dt.date
    return (_finalize(df[d < cut].drop(columns='serial_number')),
            _finalize(df[d >= cut].drop(columns='serial_number')))


def median_spread(df):
    s = df['spread'][df['spread'] > 0] if 'spread' in df else pd.Series(dtype=float)
    return float(s.median()) if len(s) else 0.0


# ---- zapis / odczyt ----------------------------------------------------------------------

def save_dataset(df, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.endswith('.h5'):
        df.to_hdf(path, key='data', mode='w')
    else:
        df.to_csv(path, index=False)


def load_dataset(path):
    if path.endswith('.h5'):
        df = pd.read_hdf(path)
    else:
        df = pd.read_csv(path, parse_dates=['datetime'])
    if 'price' not in df.columns and 'Price' in df.columns:  # przykladowe dane SGXTW z TradingGym
        df = df.rename(columns={'Price': 'price', 'Volume': 'volume'})
        df['spread'] = (df['Ask_price'] - df['Bid_price']).abs() if 'Ask_price' in df else 0.0
        df = df[['datetime', 'price', 'spread', 'volume']]
    df['datetime'] = pd.to_datetime(df['datetime'])
    return _finalize(df.drop(columns=[c for c in ('serial_number',) if c in df.columns]))
