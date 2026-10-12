"""Trening agenta RL (stable-baselines3) na danych z MT5, z wykorzystaniem wszystkich zasobow."""
import json
import logging
import os
from functools import partial

from .config import AgentConfig
from .gym_env import RandomPolicy, RLPolicy, TradingGymEnv, run_backtest
from .hardware import configure_torch
from .mt5_data import split_days

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


def train(df, cfg, timesteps, out_dir, profile, test_frac=0.2, seed=0, n_envs=None, checkpoint_every=200_000):
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    configure_torch(profile)
    train_df, test_df = split_days(df, test_frac)
    n_envs = n_envs or profile.n_envs
    log.info('Trening %s: %d dni treningowych, %d testowych, %d srodowisk rownolegle, urzadzenie %s',
             cfg.algo, train_df['datetime'].dt.date.nunique(), test_df['datetime'].dt.date.nunique(),
             n_envs, profile.device)
    fns = [partial(_make_env, train_df, cfg, seed + i) for i in range(n_envs)]
    vec = VecMonitor(SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns))
    try:
        model = build_model(cfg, vec, profile, timesteps, seed=seed)
        ckpt = CheckpointCallback(save_freq=max(1, checkpoint_every // n_envs),
                                  save_path=os.path.join(out_dir, 'checkpoints'), name_prefix='model')
        model.learn(total_timesteps=timesteps, callback=ckpt, progress_bar=False)
    finally:
        vec.close()
    cfg.save(out_dir)
    model.save(os.path.join(out_dir, 'model.zip'))
    log.info('Zapisano model do %s', out_dir)
    return model, evaluate(model, cfg, train_df, test_df, out_dir)


def evaluate(model, cfg, train_df, test_df, out_dir=None):
    """Backtest na danych testowych (nieznanych modelowi) + porownanie z losowym agentem."""
    results = {
        'rl_test': run_backtest(test_df, cfg, RLPolicy(model))[0],
        'random_test': run_backtest(test_df, cfg, RandomPolicy())[0],
        'rl_train': run_backtest(train_df, cfg, RLPolicy(model))[0],
    }
    if out_dir:
        with open(os.path.join(out_dir, 'backtest.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    return results


def load_model(model_dir, device='auto'):
    cfg = AgentConfig.load(model_dir)
    return _algo_cls(cfg.algo).load(os.path.join(model_dir, 'model.zip'), device=device), cfg
