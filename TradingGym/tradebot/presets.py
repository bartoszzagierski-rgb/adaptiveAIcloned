"""Gotowe ustawienia dla instrumentow. Uzycie: python -m tradebot fetch --preset xauusd ...

XAUUSD (zloto):
  * swiece M5 zamiast tickow - lata historii, mniej szumu (zloto ma setki tysiecy tickow dziennie),
  * okno 48 swiec = 4 h kontekstu, decyzja co swiece (5 min),
  * cechy: cena, spread, wolumen, zakres swiecy (high-low),
  * historia: 10 lat (albo maksimum dostepne u brokera),
  * koszt = 1.5 x mediana spreadu z ostatniego roku, liczona jako % ceny - przez 10 lat cena zlota wzrosla
    ok. 3,5 raza, wiec staly koszt w USD bylby za duzy dla starych lat,
  * live: nie otwiera pozycji gdy spread > 3 x mediana (jako % ceny; newsy, rollover),
  * sesja: zloto ma przerwe ok. polnocy czasu serwera. W treningu kazdy dzien to osobny epizod, pierwsze
    48 swiec (4 h od otwarcia ok. 01:00) to tylko obserwacja, a pozycja jest zamykana na koniec dnia.
    Na zywo odwzorowujemy to samo: handel od 05:00, zamkniecie pozycji po 23:30 (czas serwera MT5).
"""

PRESETS = {
    'xauusd': {
        'symbol': 'XAUUSD',
        'source': 'bars', 'timeframe': 'M5', 'days': 3650,  # 10 lat albo tyle, ile ma broker
        'features': ['price', 'spread', 'volume', 'range'],
        'obs_len': 48, 'step_len': 1, 'max_position': 3,
        'fee_mult': 1.5, 'max_spread_mult': 3.0,
        'session_start': '05:00', 'flat_after': '23:30',
        # trening do skutku: sprawdzenie na walidacji co 2M krokow, koniec po 10 sprawdzeniach bez poprawy
        # (min. ~20M krokow); bezpieczniki: 300M krokow albo 72 h
        'max_timesteps': 300_000_000, 'eval_every': 2_000_000, 'patience': 10, 'max_hours': 72,
        'lot': 0.01, 'max_daily_loss_pct': 2.0, 'max_drawdown_pct': 10.0,
        # douczanie (retrain): ostatnie 180 dni, 10 ostatnich dni jako sprawdzian
        'retrain_days': 180, 'retrain_holdout_days': 10, 'retrain_timesteps': 500_000,
    },
}


def get(name):
    if not name:
        return {}
    try:
        return PRESETS[name.lower()]
    except KeyError:
        raise SystemExit('Nieznany preset {} (dostepne: {})'.format(name, ', '.join(PRESETS)))
