"""Trening agenta RL (stable-baselines3) na danych z MT5, z wykorzystaniem wszystkich zasobow."""
import json
import logging
import os
import time
from functools import partial

import numpy as np

from .config import AgentConfig
from .gym_env import RandomPolicy, RLPolicy, TradingGymEnv, run_backtest
from .hardware import configure_torch
from .mt5_data import split_days, split_periods

log = logging.getLogger(__name__)


def _make_env(df, cfg, seed):
    env = TradingGymEnv(df, cfg)
    env.reset(seed=seed)
    return env


def _algo_cls(name):
    from stable_baselines3 import DQN, PPO
    return {'PPO': PPO, 'DQN': DQN}[name.upper()]


def build_model(cfg, vec_env, profile, timesteps, seed=0, verbose=1):
    n_envs = vec_env.num_envs
    policy_kwargs = dict(net_arch=profile.net_arch)
    common = dict(policy='MlpPolicy', env=vec_env, seed=seed, verbose=verbose, device=profile.device,
                  policy_kwargs=policy_kwargs)
    if cfg.algo.upper() == 'PPO':
        n_steps = int(min(1024, max(64, timesteps // n_envs)))
        return _algo_cls('PPO')(n_steps=n_steps, batch_size=int(min(profile.batch_size, n_steps * n_envs)),
                                ent_coef=0.01, learning_rate=3e-4, gamma=0.99, **common)
    obs_dim = vec_env.observation_space.shape[0]
    return _algo_cls('DQN')(buffer_size=profile.dqn_buffer_size(obs_dim),
                            learning_starts=int(min(10_000, max(100, timesteps // 10))),
                            batch_size=int(min(1024, profile.batch_size)), train_freq=4, gradient_steps=max(1, n_envs // 4),
                            exploration_fraction=0.2, target_update_interval=2000, **common)


class LearnUntilNoProgress:
    """Callback: co `eval_every` krokow sprawdza model na walidacji (dni, ktorych nie uczy sie),
    zapisuje najlepszy i zatrzymuje trening, gdy przez `patience` kolejnych sprawdzen nie bylo poprawy
    (albo minal `max_hours`). Ocena = Sharpe dziennych wynikow (zysk wzgledem zmiennosci)."""

    def __init__(self, val_df, cfg, out_dir, eval_every, patience, max_hours=None):
        from stable_baselines3.common.callbacks import BaseCallback

        outer = self

        class _CB(BaseCallback):
            def _on_step(self):
                return outer.on_step(self.model, self.num_timesteps)

        self.callback = _CB()
        self.val_df, self.cfg, self.out_dir = val_df, cfg, out_dir
        self.eval_every, self.patience = eval_every, patience
        self.deadline = None if not max_hours else time.time() + max_hours * 3600
        self.next_eval = eval_every
        self.best_score, self.best_step, self.bad = -np.inf, 0, 0
        self.history = []
        self.stop_reason = 'osiagnieto limit krokow'

    def on_step(self, model, step):
        if self.deadline and time.time() > self.deadline:
            self.stop_reason = 'minal limit czasu'
            return False
        if step < self.next_eval:
            return True
        self.next_eval += self.eval_every
        s = run_backtest(self.val_df, self.cfg, RLPolicy(model))[0]
        score = s['sharpe'] if s['trades'] >= 10 else -1.0  # model, ktory nie handluje, to nie postep
        improved = bool(score > self.best_score + 1e-3)
        if improved:
            self.best_score, self.best_step, self.bad = score, step, 0
            model.save(os.path.join(self.out_dir, 'model.zip'))
        else:
            self.bad += 1
        self.history.append({'step': step, 'sharpe': s['sharpe'], 'pnl_bps': s['pnl_bps'], 'trades': s['trades'],
                             'win_rate': s['win_rate'], 'best': improved})
        with open(os.path.join(self.out_dir, 'progress.json'), 'w', encoding='utf-8') as f:
            json.dump({'best_step': self.best_step, 'best_sharpe': self.best_score, 'history': self.history}, f,
                      indent=2)
        log.info('[walidacja] krok %s: sharpe %.2f, wynik %+.0f bps, transakcji %d -> %s (bez poprawy: %d/%d)',
                 '{:,}'.format(step), s['sharpe'], s['pnl_bps'], s['trades'],
                 'NAJLEPSZY, zapisano' if improved else 'bez poprawy', self.bad, self.patience)
        if self.bad >= self.patience:
            self.stop_reason = 'brak poprawy przez {} sprawdzen - model sie nauczyl'.format(self.patience)
            return False
        return True


def train(df, cfg, timesteps, out_dir, profile, test_months=6, val_months=6, seed=0, n_envs=None,
          checkpoint_every=1_000_000, eval_every=None, patience=10, max_hours=None):
    """Trening do skutku: uczy sie, dopoki wynik na walidacji rosnie (max `timesteps` krokow / `max_hours`),
    potem egzamin na ostatnich `test_months` miesiacach, ktorych model nigdy nie widzial."""
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    configure_torch(profile)
    os.makedirs(out_dir, exist_ok=True)
    cfg.save(out_dir)
    train_df, val_df, test_df = split_periods(df, test_months, val_months)
    n_envs = n_envs or profile.n_envs
    eval_every = eval_every or max(n_envs, min(1_000_000, timesteps // 10))

    def span(x):
        return '{} dni ({} .. {})'.format(x['datetime'].dt.date.nunique(), x['datetime'].iloc[0].date(),
                                         x['datetime'].iloc[-1].date())
    log.info('Trening %s: %s | walidacja %s | EGZAMIN %s', cfg.algo, span(train_df), span(val_df), span(test_df))
    log.info('%d srodowisk rownolegle, urzadzenie %s, sprawdzenie co %s krokow, stop po %d sprawdzeniach bez '
             'poprawy, max %s krokow%s', n_envs, profile.device, '{:,}'.format(eval_every), patience,
             '{:,}'.format(timesteps), ', max {} h'.format(max_hours) if max_hours else '')
    fns = [partial(_make_env, train_df, cfg, seed + i) for i in range(n_envs)]
    vec = VecMonitor(SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns))
    stopper = LearnUntilNoProgress(val_df, cfg, out_dir, eval_every, patience, max_hours)
    t0 = time.time()
    try:
        model = build_model(cfg, vec, profile, timesteps, seed=seed)
        ckpt = CheckpointCallback(save_freq=max(1, checkpoint_every // n_envs),
                                  save_path=os.path.join(out_dir, 'checkpoints'), name_prefix='model')
        try:
            model.learn(total_timesteps=timesteps, callback=CallbackList([ckpt, stopper.callback]),
                        progress_bar=False)
        except KeyboardInterrupt:
            stopper.stop_reason = 'przerwano Ctrl+C'
            log.warning('Przerwano - przechodze do egzaminu z najlepszym dotad modelem.')
    finally:
        vec.close()
    best_path = os.path.join(out_dir, 'model.zip')
    if not os.path.exists(best_path):  # zadnego sprawdzenia (bardzo krotki trening) -> ostatni model
        model.save(best_path)
    best = _algo_cls(cfg.algo).load(best_path, device='cpu')
    log.info('Koniec nauki: %s. Czas %.1f h, najlepszy model z kroku %s.', stopper.stop_reason,
             (time.time() - t0) / 3600, '{:,}'.format(stopper.best_step))
    return best, evaluate(best, cfg, train_df, val_df, test_df, out_dir,
                          info={'stop_reason': stopper.stop_reason, 'best_step': stopper.best_step,
                                'hours': round((time.time() - t0) / 3600, 2)})


def exam_verdict(rl, rnd):
    """Czy agent 'umie' - ocena wyniku na egzaminie (dane, ktorych nigdy nie widzial)."""
    if rl['trades'] < 20:
        return 'NIE ZDAL', ('agent prawie nie handluje ({} transakcji) - nauczyl sie, ze najbezpieczniej nic nie '
                            'robic. Sprobuj innego --step-len albo mniejszego kosztu.'.format(rl['trades']))
    if rl['pnl_bps'] > 0 and rl['sharpe'] >= 0.5 and rl['pnl_bps'] > rnd['pnl_bps']:
        return 'ZDAL', ('zarabia na nowych danych: {:+.0f} bps, sharpe {:.2f}, {}/{} dni na plus. Nastepny krok: '
                        'dry-run, potem kilka tygodni na demo.'.format(rl['pnl_bps'], rl['sharpe'],
                                                                       rl['positive_days'], rl['days']))
    if rl['pnl_bps'] > rnd['pnl_bps']:
        return 'SLABO', ('lepszy od losowego ({:+.0f} vs {:+.0f} bps), ale nie zarabia stabilnie (sharpe {:.2f}). '
                         'Nie nadaje sie jeszcze do handlu.'.format(rl['pnl_bps'], rnd['pnl_bps'], rl['sharpe']))
    return 'NIE ZDAL', 'nie lepszy od losowego ({:+.0f} vs {:+.0f} bps).'.format(rl['pnl_bps'], rnd['pnl_bps'])


def evaluate(model, cfg, train_df, val_df, test_df, out_dir=None, info=None):
    """Egzamin na danych testowych (nieznanych modelowi) + porownanie z losowym agentem."""
    days = sorted(train_df['datetime'].dt.date.unique())
    recent_train = train_df[train_df['datetime'].dt.date.isin(days[-125:])]  # ~6 mies. treningu (por. przeuczenia)
    results = {
        'rl_test': run_backtest(test_df, cfg, RLPolicy(model))[0],
        'random_test': run_backtest(test_df, cfg, RandomPolicy())[0],
        'rl_val': run_backtest(val_df, cfg, RLPolicy(model))[0],
        'rl_train_last_6m': run_backtest(recent_train, cfg, RLPolicy(model))[0],
    }
    grade, why = exam_verdict(results['rl_test'], results['random_test'])
    results['exam'] = {'verdict': grade, 'reason': why}
    if info:
        results['training'] = info
    if out_dir:
        with open(os.path.join(out_dir, 'backtest.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    log.info('EGZAMIN (%s): %s - %s', results['rl_test']['period'], grade, why)
    return results


def load_model(model_dir, device='auto'):
    cfg = AgentConfig.load(model_dir)
    return _algo_cls(cfg.algo).load(os.path.join(model_dir, 'model.zip'), device=device), cfg


def retrain(model_dir, df, profile, timesteps, holdout_days=10, n_envs=None, seed=0, min_gain_bps=0.0):
    """Douczanie typu champion/challenger (walk-forward).

    1. Ostatnie `holdout_days` dni odkladamy jako sprawdzian - model ich nie widzi.
    2. Challenger = kopia obecnego modelu douczona na swiezych danych (reszta df).
    3. Oba modele + agent losowy graja na dniach sprawdzianu.
    4. Challenger zastepuje obecny model tylko gdy jest lepszy o >= min_gain_bps i lepszy od losowego.
       Stary model trafia do history/ (mozna wrocic).
    """
    import shutil
    import time as _time
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    configure_torch(profile)
    champion, cfg = load_model(model_dir, device=profile.device)
    days = sorted(df['datetime'].dt.date.unique())
    if len(days) <= holdout_days + 5:
        raise ValueError('Za malo dni ({}) do douczania z {} dniami sprawdzianu'.format(len(days), holdout_days))
    train_df, test_df = split_days(df, holdout_days / len(days))
    n_envs = n_envs or profile.n_envs
    fns = [partial(_make_env, train_df, cfg, seed + i) for i in range(n_envs)]
    vec = VecMonitor(SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns))
    try:
        challenger = _algo_cls(cfg.algo).load(os.path.join(model_dir, 'model.zip'), env=vec, device=profile.device)
        challenger.learn(total_timesteps=timesteps, reset_num_timesteps=False, progress_bar=False)
    finally:
        vec.close()
    res = {
        'champion': run_backtest(test_df, cfg, RLPolicy(champion))[0],
        'challenger': run_backtest(test_df, cfg, RLPolicy(challenger))[0],
        'random': run_backtest(test_df, cfg, RandomPolicy())[0],
    }
    gain = res['challenger']['pnl_bps'] - res['champion']['pnl_bps']
    promote = gain >= min_gain_bps and res['challenger']['pnl_bps'] > res['random']['pnl_bps']
    stamp = _time.strftime('%Y%m%d-%H%M%S')
    if promote:
        hist = os.path.join(model_dir, 'history')
        os.makedirs(hist, exist_ok=True)
        shutil.copy2(os.path.join(model_dir, 'model.zip'), os.path.join(hist, 'model-{}.zip'.format(stamp)))
        tmp = os.path.join(model_dir, 'model.new.zip')
        challenger.save(tmp)
        os.replace(tmp, os.path.join(model_dir, 'model.zip'))  # atomowo - live przeladuje nowy model
    entry = {'time': stamp, 'train_days': len(days) - holdout_days, 'holdout_days': holdout_days,
             'timesteps': timesteps, 'gain_bps': round(gain, 2), 'promoted': promote, 'results': res}
    path = os.path.join(model_dir, 'retrain_log.json')
    hist_log = []
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            hist_log = json.load(f)
    hist_log.append(entry)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(hist_log, f, indent=2)
    log.info('Douczanie: challenger %+.1f bps vs champion %+.1f bps (losowy %+.1f) -> %s',
             res['challenger']['pnl_bps'], res['champion']['pnl_bps'], res['random']['pnl_bps'],
             'NOWY MODEL WDROZONY' if promote else 'zostaje obecny model')
    return entry
