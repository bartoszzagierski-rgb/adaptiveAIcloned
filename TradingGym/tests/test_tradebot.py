"""Testy bez prawdziwego MT5 i Ollamy (symulator FakeMT5 + falszywy klient LLM)."""
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from tradebot import mt5_data
from tradebot.config import AgentConfig
from tradebot.fake_mt5 import ACCOUNT_TRADE_MODE_REAL, FakeMT5, default_fake
from tradebot.features import make_obs, obs_dim
from tradebot.gym_env import RandomPolicy, TradingGymEnv, run_backtest
from tradebot.live import LiveError, LiveTrader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


@pytest.fixture
def fake():
    f = default_fake()
    mt5_data.use_module(f)
    yield f
    mt5_data.use_module(None)


@pytest.fixture
def tick_df(fake):
    end = datetime.fromtimestamp(fake.now(), tz=timezone.utc)
    raw = mt5_data.fetch_ticks('EURUSD', end - timedelta(hours=3), end)
    # 3 godziny -> sztucznie dzielimy na 3 "dni" zeby miec kilka epizodow
    df = mt5_data.ticks_to_gym(raw)
    df['datetime'] = df['datetime'] + pd.to_timedelta(np.minimum(np.arange(len(df)) // (len(df) // 3), 2), unit='D')
    return mt5_data._finalize(df.drop(columns='serial_number'))


def cfg(**kw):
    base = dict(symbol='EURUSD', obs_len=32, step_len=8, max_position=2, fee=0.0001)
    base.update(kw)
    return AgentConfig(**base)


def test_ticks_to_gym_format(tick_df):
    df = tick_df
    assert list(df.index) == list(range(len(df)))
    assert {'datetime', 'price', 'spread', 'volume', 'serial_number'} <= set(df.columns)
    assert (df.groupby(df['datetime'].dt.date)['serial_number'].min() == 0).all()
    assert np.allclose(df['price'], (df['bid'] + df['ask']) / 2)


def test_bars_to_gym(fake):
    end = datetime.fromtimestamp(fake.now(), tz=timezone.utc)
    raw, point = mt5_data.fetch_bars('EURUSD', 'M5', end - timedelta(hours=3), end)
    df = mt5_data.bars_to_gym(raw, point)
    assert len(df) >= 30 and (df['spread'] > 0).all()


def test_sample_dataset_loads():
    df = mt5_data.load_dataset(os.path.join(ROOT, 'trading_env/test/data/SGXTWsample.csv'))
    assert {'price', 'spread', 'volume', 'serial_number'} <= set(df.columns)


def test_obs_shape_and_clip():
    w = np.column_stack([np.linspace(1, 1.1, 32), np.ones(32), np.arange(32)])
    o = make_obs(w, 1, 2, 0.01)
    assert o.shape == (obs_dim(3, 32),) and np.abs(o).max() <= 10


def test_gym_env_passes_checker(tick_df):
    from gymnasium.utils.env_checker import check_env
    check_env(TradingGymEnv(tick_df, cfg()), skip_render_check=True)


def test_episode_reward_equals_realized_pnl(tick_df):
    env = TradingGymEnv(tick_df, cfg())
    env.reset(seed=3)
    rng = np.random.default_rng(0)
    total, done = 0.0, False
    while not done:
        _, r, done, _, info = env.step(rng.integers(3))
        total += r
    tr = info['transactions']
    realized_bps = (tr['reward'] / tr['price']).sum() * 1e4
    assert total == pytest.approx(realized_bps, rel=0.05, abs=0.5)


def test_backtest_runs_all_days(tick_df):
    summary, tr = run_backtest(tick_df, cfg(), RandomPolicy())
    assert summary['days'] == 3 and summary['trades'] > 0


def test_train_ppo_and_dqn_smoke(tick_df, tmp_path):
    from tradebot.hardware import detect
    from tradebot.train import load_model, train
    prof = detect()
    prof.cuda = False
    for algo in ('PPO', 'DQN'):
        out = str(tmp_path / algo)
        _, res = train(tick_df, cfg(algo=algo), 512, out, prof, n_envs=2)
        assert {'rl_test', 'random_test', 'rl_val', 'rl_train_last_6m', 'exam', 'training'} <= set(res)
        model, c = load_model(out, device='cpu')
        assert c.algo == algo and os.path.exists(os.path.join(out, 'backtest.json'))


class AlwaysBuy:
    name = 'buy'

    def act(self, obs, window, position, unrealized):
        return 1


class AlwaysSell(AlwaysBuy):
    def act(self, obs, window, position, unrealized):
        return 2


def _trader(fake, policy, tmp_path, **kw):
    t = LiveTrader(cfg(), policy, fake, log_path=str(tmp_path / 'log.csv'), stop_file=str(tmp_path / 'STOP'), **kw)
    t.start()
    return t


def _steps(fake, trader, n):
    recs = []
    for _ in range(n):
        fake.advance(5)
        r = trader.iteration()
        if r:
            recs.append(r)
    return recs


def test_live_demo_respects_max_position_and_closes(fake, tmp_path):
    t = _trader(fake, AlwaysBuy(), tmp_path)
    _steps(fake, t, 10)
    units, _ = t.position_state(1.1)
    assert units == pytest.approx(2)  # max_position
    assert all(o['magic'] == t.magic for o in fake.orders)
    t.policy = AlwaysSell()
    _steps(fake, t, 10)
    units, _ = t.position_state(1.1)
    assert units == pytest.approx(-2)  # zamknal 2 longi i otworzyl 2 shorty
    assert any(o.get('position') for o in fake.orders)
    assert os.path.exists(tmp_path / 'log.csv')


def test_dry_run_sends_no_orders(fake, tmp_path):
    t = _trader(fake, AlwaysBuy(), tmp_path, dry_run=True)
    _steps(fake, t, 6)
    assert fake.orders == [] and t.position_state(1.1)[0] > 0


def test_demo_mode_refuses_real_account(tmp_path):
    f = FakeMT5(trade_mode=ACCOUNT_TRADE_MODE_REAL, now=default_fake().now())
    with pytest.raises(LiveError, match='REALNE'):
        LiveTrader(cfg(), AlwaysBuy(), f, log_path=None).start()


def test_live_mode_requires_real_account_confirmation_and_loss_limit(fake):
    with pytest.raises(LiveError):
        LiveTrader(cfg(), AlwaysBuy(), fake, mode='live', confirm_live=True, max_daily_loss=10).start()  # demo konto
    real = FakeMT5(trade_mode=ACCOUNT_TRADE_MODE_REAL, now=fake.now())
    with pytest.raises(LiveError, match='potwierdzony'):
        LiveTrader(cfg(), AlwaysBuy(), real, mode='live', max_daily_loss=10).start()
    with pytest.raises(LiveError, match='max-daily-loss'):
        LiveTrader(cfg(), AlwaysBuy(), real, mode='live', confirm_live=True).start()


def test_stop_file_and_daily_loss(fake, tmp_path):
    t = _trader(fake, AlwaysBuy(), tmp_path, max_daily_loss=1.0)
    _steps(fake, t, 4)
    fake.balance -= 5  # symulowana strata
    assert t.iteration() is None and not t.running and t.position_state(1.1)[0] == 0
    t2 = _trader(fake, AlwaysBuy(), tmp_path)
    open(tmp_path / 'STOP', 'w').close()
    t2.iteration()
    assert not t2.running


def test_bar_feed_live(fake, tmp_path):
    t = LiveTrader(cfg(source='bars', timeframe='M1', step_len=1), AlwaysBuy(), fake, log_path=None,
                   stop_file=None)
    t.start()
    fake.advance(61)
    assert t.iteration() is not None


class FakeLLM:
    model = 'fake'

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def chat_json(self, system, user):
        self.calls.append(json.loads(user))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def test_llm_policy_and_supervisor(fake, tmp_path):
    from tradebot.llm import LLMPolicy, LLMSupervisor
    w = np.column_stack([np.linspace(1, 1.01, 32), np.full(32, 1e-4), np.zeros(32)])
    c = cfg()
    assert LLMPolicy(FakeLLM({'action': 'sell'}), c).act(None, w, 0, 0) == 2
    assert LLMPolicy(FakeLLM(RuntimeError('down')), c).act(None, w, 0, 0) == 0
    assert LLMSupervisor(FakeLLM({'approve': True}), c).review(1, w, 0, 0)[0]
    assert not LLMSupervisor(FakeLLM(RuntimeError('down')), c).review(1, w, 0, 0)[0]
    llm = FakeLLM({'approve': False, 'reason': 'too volatile'})
    t = _trader(fake, AlwaysBuy(), tmp_path, supervisor=LLMSupervisor(llm, c))
    recs = _steps(fake, t, 4)
    assert recs and all(r['result'] == 'vetoed by LLM' for r in recs) and fake.orders == []
    assert 'return_bps' in llm.calls[0]['market']


def test_hardware_profile_picks_model_for_rtx5080():
    from tradebot.hardware import HardwareProfile
    t7 = HardwareProfile(cpu_threads=20, ram_gb=64, gpu_name='NVIDIA GeForce RTX 5080', vram_gb=15.9, cuda=True,
                         free_disk_gb=1800)
    assert t7.llm_model == 'qwen3:14b' and t7.n_envs == 18 and t7.device == 'cuda'
    assert t7.dqn_buffer_size(obs_dim(3, 64)) > 1_000_000


def test_cli_fake_pipeline(tmp_path, monkeypatch):
    from tradebot.__main__ import main
    monkeypatch.chdir(tmp_path)
    data = str(tmp_path / 'd.csv')
    main(['fetch', '--fake-mt5', '--symbol', 'EURUSD', '--source', 'bars', '--timeframe', 'M1', '--days', '3',
          '--out', data])
    main(['train', '--data', data, '--out', 'm', '--timesteps', '256', '--n-envs', '1', '--device', 'cpu',
          '--obs-len', '32', '--step-len', '4'])
    main(['backtest', '--model', 'm', '--data', data, '--policy', 'random'])
    main(['live', '--fake-mt5', '--model', 'm', '--dry-run', '--max-iterations', '20', '--poll', '0'])
    assert os.path.exists('live_trades.csv')
    mt5_data.use_module(None)


def test_resolve_symbol_broker_suffix(fake):
    assert mt5_data.resolve_symbol('EURUSD') == 'EURUSD'
    assert mt5_data.resolve_symbol('XAUUSD') == 'XAUUSDm'
    with pytest.raises(RuntimeError, match='niedostepny'):
        mt5_data.resolve_symbol('BTCXYZ')


def test_session_flat_and_start(fake, tmp_path):
    c = cfg(session_start='13:00', flat_after='12:30')
    t = LiveTrader(c, AlwaysBuy(), fake, log_path=None, stop_file=None)
    t.start()
    assert not t.in_session()  # 12:00 < 13:00
    c.session_start = '10:00'
    c.flat_after = '23:00'
    recs = _steps(fake, t, 4)
    assert recs and t.position_state(1.1)[0] > 0
    c.flat_after = '11:00'  # po godzinie zamkniecia -> zamyka pozycje i nie handluje
    assert t.iteration() is None and t.position_state(1.1)[0] == 0


def test_cli_xauusd_preset_pipeline(tmp_path, monkeypatch):
    from tradebot.__main__ import main
    from tradebot.config import AgentConfig
    monkeypatch.chdir(tmp_path)
    main(['fetch', '--fake-mt5', '--preset', 'xauusd', '--days', '4', '--out', 'gold.csv'])
    meta = json.load(open('gold.csv.meta.json'))
    assert meta['symbol'] == 'XAUUSDm' and meta['source'] == 'bars' and meta['timeframe'] == 'M5'
    main(['train', '--data', 'gold.csv', '--out', 'g', '--timesteps', '256', '--n-envs', '1', '--device', 'cpu'])
    c = AgentConfig.load('g')
    assert c.features == ['price', 'spread', 'volume', 'range'] and c.obs_len == 48 and c.step_len == 1
    assert c.fee_bps > 0 and c.max_spread_bps == pytest.approx(c.fee_bps * 2) and c.flat_after == '23:30'
    assert c.fee_at(4000) == pytest.approx(c.fee_bps * 0.4)
    main(['live', '--fake-mt5', '--model', 'g', '--dry-run', '--max-iterations', '10', '--poll', '0'])
    mt5_data.use_module(None)


def test_pct_limits_and_margin_with_10k_pln(tmp_path):
    f = FakeMT5(base_price=2400.0, now=default_fake().now())
    f.currency, f.balance = 'PLN', 10000.0
    mt5_data.use_module(f)
    c = cfg(symbol='XAUUSDm')
    t = LiveTrader(c, AlwaysBuy(), f, log_path=None, stop_file=None, max_daily_loss_pct=2.0,
                   max_drawdown_pct=10.0)
    t.start()
    assert t.daily_limit() == pytest.approx(200.0)  # 2% z 10 000 zl
    _steps(f, t, 3)
    assert t.position_state(2400)[0] > 0
    f.balance -= 250  # strata > 200 zl -> stop i zamkniecie
    assert t.iteration() is None and not t.running and t.position_state(2400)[0] == 0
    # depozyt: przy malym wolnym depozycie bot nie otwiera pozycji
    t2 = LiveTrader(c, AlwaysBuy(), f, log_path=None, stop_file=None, max_margin_use=0.0001)
    t2.start()
    recs = _steps(f, t2, 3)
    assert recs and all(r['result'].startswith('skip: depozyt') for r in recs)
    mt5_data.use_module(None)


def test_retrain_champion_challenger_and_live_reload(tmp_path, tick_df, monkeypatch):
    from tradebot import train as tr
    from tradebot.hardware import detect
    prof = detect()
    prof.cuda = False
    out = str(tmp_path / 'm')
    tr.train(tick_df, cfg(), 256, out, prof, n_envs=1)
    days = tick_df['datetime'].dt.date.unique()
    more = tick_df.copy()
    for k in range(1, 3):  # 9 dni danych: 3 oryginalne x3
        extra = tick_df.copy()
        extra['datetime'] += pd.Timedelta(days=3 * k)
        more = pd.concat([more, extra])
    more = mt5_data._finalize(more.drop(columns='serial_number').reset_index(drop=True))
    with pytest.raises(ValueError):
        tr.retrain(out, tick_df, prof, 128, holdout_days=2, n_envs=1)
    e = tr.retrain(out, more, prof, 128, holdout_days=2, n_envs=1, min_gain_bps=-1e9)
    assert set(e['results']) == {'champion', 'challenger', 'random'}
    log = json.load(open(os.path.join(out, 'retrain_log.json')))
    assert len(log) == 1
    if e['promoted']:
        assert os.listdir(os.path.join(out, 'history'))
    e2 = tr.retrain(out, more, prof, 128, holdout_days=2, n_envs=1, min_gain_bps=1e9)
    assert not e2['promoted']


def test_train_stops_when_no_progress_and_exam(tick_df, tmp_path):
    from tradebot.hardware import detect
    from tradebot.train import train
    prof = detect()
    prof.cuda = False
    out = str(tmp_path / 'es')
    _, res = train(tick_df, cfg(), 1_000_000, out, prof, n_envs=1, eval_every=64, patience=2)
    prog = json.load(open(os.path.join(out, 'progress.json')))
    assert res['training']['stop_reason'].startswith('brak poprawy') and prog['history'][-1]['step'] < 1_000_000
    assert res['exam']['verdict'] in ('ZDAL', 'SLABO', 'NIE ZDAL')


def test_split_periods_months():
    from tradebot.mt5_data import _finalize, split_periods
    t = pd.date_range('2016-01-01', '2026-01-01', freq='12h')
    df = _finalize(pd.DataFrame({'datetime': t, 'price': 1.0, 'spread': 0.0, 'volume': 0.0}))
    tr, va, te = split_periods(df, 6, 6)
    assert te['datetime'].min() >= pd.Timestamp('2025-07-01') and va['datetime'].min() >= pd.Timestamp('2025-01-01')
    assert tr['datetime'].max() < va['datetime'].min() and va['datetime'].max() < te['datetime'].min()


def test_exam_verdict():
    from tradebot.train import exam_verdict
    base = dict(days=120, positive_days=70)
    assert exam_verdict(dict(base, trades=500, pnl_bps=300, sharpe=1.2), dict(trades=900, pnl_bps=-800))[0] == 'ZDAL'
    assert exam_verdict(dict(base, trades=5, pnl_bps=0, sharpe=0), dict(trades=900, pnl_bps=-800))[0] == 'NIE ZDAL'
    assert exam_verdict(dict(base, trades=500, pnl_bps=-50, sharpe=-0.1), dict(trades=900, pnl_bps=-800))[0] == 'SLABO'
