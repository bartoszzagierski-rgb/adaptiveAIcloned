"""Opakowanie TradingGym w standardowe srodowisko gymnasium (dla stable-baselines3) + backtest."""
import logging

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

import trading_env
from .features import OBS_CLIP, make_obs, obs_dim
from .mt5_data import drop_short_days

# kolumny dopisywane przez TradingGym v1 za cechami (return_transaction=True)
_POS, _PRICE, _FLUC = 0, 3, 5

logging.getLogger('training_v1').setLevel(logging.WARNING)
logging.getLogger('backtest_v1').setLevel(logging.WARNING)


class TradingGymEnv(gym.Env):
    """Akcje: 0 = nic, 1 = kup 1 jednostke (lub zamknij 1 short), 2 = sprzedaj 1 (lub zamknij 1 long).

    Nagroda (punkty bazowe ceny): wynik mark-to-market = zrealizowany zysk + zmiana niezrealizowanego.
    Suma nagrod w epizodzie = zrealizowany wynik dnia (pozycja jest zamykana na koniec dnia).
    """
    metadata = {'render_modes': []}

    def __init__(self, df, cfg, env_id='training_v1'):
        super().__init__()
        self.cfg = cfg
        self.n_feat = len(cfg.features)
        df = drop_short_days(df, cfg.obs_len + 2 * cfg.step_len + 1)
        self.env = trading_env.make(env_id=env_id, obs_data_len=cfg.obs_len, step_len=cfg.step_len, df=df,
                                    fee=cfg.fee, max_position=cfg.max_position, deal_col_name='price',
                                    feature_names=cfg.features, return_transaction=True)
        self.observation_space = spaces.Box(-OBS_CLIP, OBS_CLIP, (obs_dim(self.n_feat, cfg.obs_len),), np.float32)
        self.action_space = spaces.Discrete(3)
        self.raw = None
        self._prev_fluc = 0.0

    # widok rynku dla polityk (LLM potrzebuje surowych cen, RL - obserwacji)
    @property
    def window(self):
        return self.raw[:, :self.n_feat]

    @property
    def position(self):
        return float(self.raw[-1, self.n_feat + _POS])

    @property
    def unrealized(self):
        return float(self.raw[-1, self.n_feat + _FLUC])

    def _obs(self):
        return make_obs(self.window, self.position, self.cfg.max_position, self.unrealized)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)  # TradingGym losuje dzien przez np.random
        self.raw = self.env.reset()
        self._prev_fluc = 0.0
        return self._obs(), {}

    def step(self, action):
        self.raw, realized, done, info = self.env.step(int(action))
        fluc = 0.0 if done else self.unrealized
        price = float(self.raw[-1, self.n_feat + _PRICE])
        reward = (float(realized) + fluc - self._prev_fluc) / price * 1e4
        self._prev_fluc = fluc
        return self._obs(), reward, bool(done), False, ({'transactions': info} if done else {})


# ---- polityki ----------------------------------------------------------------------------

class RandomPolicy:
    name = 'random'

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, window, position, unrealized):
        return int(self.rng.integers(3))


class RLPolicy:
    name = 'rl'

    def __init__(self, model):
        self.model = model

    def act(self, obs, window, position, unrealized):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def run_backtest(df, cfg, policy):
    """Przechodzi po kolei przez wszystkie dni (backtest_v1). Zwraca (podsumowanie, transakcje)."""
    env = TradingGymEnv(df, cfg, env_id='backtest_v1')
    days = []
    while not env.env.backtest_done:
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, done, _, info = env.step(policy.act(obs, env.window, env.position, env.unrealized))
        days.append(info['transactions'])
    return summarize(pd.concat(days), cfg, policy.name), pd.concat(days)


def summarize(tr, cfg, name):
    tr = tr.copy()
    tr['day'] = pd.to_datetime(tr['datetime']).dt.date
    closes = tr[tr['reward_makereal'] != 0]
    daily = tr.groupby('day')['reward'].sum()
    mean_price = float(tr['price'].mean())
    pnl = float(tr['reward'].sum())
    return {
        'policy': name,
        'days': int(len(daily)),
        'pnl_price_units': round(pnl, 6),
        'pnl_bps': round(pnl / mean_price * 1e4, 2),
        'trades': int((tr['position_variation'] != 0).sum()),
        'closing_trades': int(len(closes)),
        'win_rate': round(float((closes['reward'] > 0).mean()), 3) if len(closes) else None,
        'best_day_bps': round(float(daily.max()) / mean_price * 1e4, 2),
        'worst_day_bps': round(float(daily.min()) / mean_price * 1e4, 2),
        'positive_days': int((daily > 0).sum()),
        'fee_per_unit': cfg.fee,
    }
