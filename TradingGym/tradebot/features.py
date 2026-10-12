"""Wspolna zamiana okna danych na obserwacje dla sieci.

Uzywana zarowno w treningu (gym_env.py), jak i na zywo (live.py), zeby agent
widzial dokladnie to samo w obu przypadkach.
"""
import numpy as np

OBS_CLIP = 10.0


def obs_dim(n_features, obs_len):
    return n_features * obs_len + 2


def make_obs(window, position, max_position, unrealized):
    """window: (obs_len, n_features), kolumna 0 = cena.
    position: pozycja w jednostkach (+long / -short), unrealized: niezrealizowany wynik w jednostkach ceny.
    """
    window = np.asarray(window, dtype=np.float64)
    price = window[:, 0]
    last = price[-1] if price[-1] > 0 else 1.0
    # cena -> odchylenie od ostatniej ceny w punktach bazowych / 10 (0.1% = 1.0)
    cols = [(price / last - 1.0) * 1e3]
    for i in range(1, window.shape[1]):
        col = window[:, i]
        std = col.std()
        cols.append((col - col.mean()) / std if std > 1e-12 else np.zeros_like(col))
    feats = np.stack(cols, axis=1).ravel()
    extra = np.array([position / max(max_position, 1),
                      unrealized / last * 1e3 / max(abs(position), 1.0)])
    return np.clip(np.concatenate([feats, extra]), -OBS_CLIP, OBS_CLIP).astype(np.float32)


def market_summary(window, features, position, max_position, unrealized):
    """Krotkie, czytelne podsumowanie rynku dla LLM (liczby zamiast surowych tickow)."""
    window = np.asarray(window, dtype=np.float64)
    price = window[:, 0]
    last = float(price[-1])

    def ret(k):
        k = min(k, len(price) - 1)
        return round((last / price[-1 - k] - 1) * 1e4, 2) if k > 0 else 0.0

    log_ret = np.diff(np.log(price)) if len(price) > 1 else np.zeros(1)
    summary = {
        'last_price': last,
        'return_bps': {'last_5_steps': ret(5), 'last_20_steps': ret(20), 'whole_window': ret(len(price) - 1)},
        'volatility_bps_per_step': round(float(log_ret.std() * 1e4), 3),
        'window_high': float(price.max()),
        'window_low': float(price.min()),
        'position_units': float(position),
        'max_position_units': int(max_position),
        'unrealized_pnl_bps': round(unrealized / last * 1e4, 2) if last else 0.0,
    }
    if 'spread' in features:
        summary['spread_bps'] = round(float(window[-1, features.index('spread')]) / last * 1e4, 3)
    return summary
