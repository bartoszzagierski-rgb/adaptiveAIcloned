"""CLI: python -m tradebot <komenda> --help

  hw        - wykryty sprzet i dobrane ustawienia (n_envs, GPU, model LLM)
  fetch     - pobierz dane z MT5 (ticki albo swiece) do pliku
  train     - trenuj agenta RL (PPO/DQN) na pobranych danych
  backtest  - sprawdz agenta (rl / random / llm) na danych
  llm-check - sprawdz czy Ollama dziala i ma model
  review    - LLM ocenia wyniki backtestu i proponuje zmiany
  live      - handel na koncie demo albo realnym (MT5)
  doctor    - sprawdz wersje: Python, pakiety, GPU/CUDA, Ollama + model, MT5

Gotowe ustawienia: --preset xauusd (fetch), dalej train/live biora je automatycznie.
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

log = logging.getLogger('tradebot')


def _mt5(args, symbol='EURUSD'):
    from . import mt5_data
    if getattr(args, 'fake_mt5', False):
        from .fake_mt5 import default_fake
        mt5_data.use_module(default_fake(symbol))
        log.warning('Uzywam SYMULATORA MT5 (--fake-mt5) - dane i zlecenia nie sa prawdziwe.')
    return mt5_data.connect()


def _llm_client(args, profile=None):
    from .hardware import detect
    from .llm import DEFAULT_HOST, OllamaClient
    profile = profile or detect()
    model = args.llm_model or profile.llm_model
    return OllamaClient(model=model, host=args.ollama_host or DEFAULT_HOST, options=profile.ollama_options())


def cmd_hw(args):
    from .hardware import detect
    print(json.dumps(detect().describe(), indent=2, ensure_ascii=False))


def cmd_fetch(args):
    from . import mt5_data, presets
    pre = presets.get(args.preset)
    args.symbol = args.symbol or pre.get('symbol')
    if not args.symbol:
        sys.exit('Podaj --symbol albo --preset')
    args.source = args.source or pre.get('source', 'ticks')
    args.timeframe = args.timeframe or pre.get('timeframe', 'M1')
    args.days = args.days or pre.get('days', 10)
    mt5 = _mt5(args, args.symbol)
    args.symbol = mt5_data.resolve_symbol(args.symbol)
    end = datetime.fromisoformat(args.end) if args.end else datetime.fromtimestamp(
        mt5.symbol_info_tick(args.symbol).time, tz=timezone.utc)
    start = datetime.fromisoformat(args.start) if args.start else end - timedelta(days=args.days)
    if args.source == 'ticks':
        df = mt5_data.ticks_to_gym(mt5_data.fetch_ticks(args.symbol, start, end), args.price_source)
    else:
        raw, point = mt5_data.fetch_bars(args.symbol, args.timeframe, start, end)
        df = mt5_data.bars_to_gym(raw, point)
    mt5_data.save_dataset(df, args.out)
    meta = {'symbol': args.symbol, 'source': args.source, 'timeframe': args.timeframe,
            'price_source': args.price_source, 'median_spread': mt5_data.median_spread(df),
            'preset': args.preset or ''}
    with open(args.out + '.meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print('Zapisano {} rekordow ({} dni) do {}'.format(len(df), df['datetime'].dt.date.nunique(), args.out))


def _load_meta(path):
    p = path + '.meta.json'
    if os.path.exists(p):
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    return {}


def cmd_train(args):
    from . import presets
    from .config import DEFAULT_FEATURES, AgentConfig
    from .hardware import detect
    from .mt5_data import load_dataset, median_spread
    from .train import train
    df = load_dataset(args.data)
    meta = _load_meta(args.data)
    preset = args.preset or meta.get('preset', '')
    pre = presets.get(preset)
    profile = detect(args.out)
    if args.device:
        profile.cuda = args.device == 'cuda'
    spread = meta.get('median_spread') or median_spread(df)
    fee = args.fee if args.fee is not None else spread * pre.get('fee_mult', 1.0)
    features = [f for f in pre.get('features', DEFAULT_FEATURES) if f in df.columns]
    cfg = AgentConfig(symbol=args.symbol or meta.get('symbol', ''), source=meta.get('source', 'ticks'),
                      timeframe=meta.get('timeframe', 'M1'), price_source=meta.get('price_source', 'mid'),
                      features=features,
                      obs_len=args.obs_len or pre.get('obs_len', 64),
                      step_len=args.step_len or pre.get('step_len', 16),
                      max_position=args.max_position or pre.get('max_position', 3),
                      fee=float(fee), algo=args.algo.upper(), preset=preset,
                      max_spread=spread * pre['max_spread_mult'] if 'max_spread_mult' in pre and spread else None,
                      session_start=pre.get('session_start'), flat_after=pre.get('flat_after'))
    args.timesteps = args.timesteps or pre.get('timesteps', 1_000_000)
    log.info('Sprzet: %s', profile.describe())
    log.info('Konfiguracja: %s', cfg)
    _, results = train(df, cfg, args.timesteps, args.out, profile, test_frac=args.test_frac, seed=args.seed,
                       n_envs=args.n_envs)
    print(json.dumps(results, indent=2))


def cmd_backtest(args):
    from .gym_env import RandomPolicy, RLPolicy, run_backtest
    from .mt5_data import load_dataset
    from .train import load_model
    model, cfg = load_model(args.model, device='cpu')
    df = load_dataset(args.data)
    if args.last_days:
        days = sorted(df['datetime'].dt.date.unique())[-args.last_days:]
        df = df[df['datetime'].dt.date.isin(days)]
    if args.policy == 'rl':
        policy = RLPolicy(model)
    elif args.policy == 'random':
        policy = RandomPolicy()
    else:
        from .llm import LLMPolicy
        client = _llm_client(args)
        problem = client.check()
        if problem:
            sys.exit(problem)
        policy = LLMPolicy(client, cfg)
    summary, tr = run_backtest(df, cfg, policy)
    if args.out:
        tr.to_csv(args.out)
    print(json.dumps(summary, indent=2))


def cmd_llm_check(args):
    client = _llm_client(args)
    problem = client.check()
    if problem:
        sys.exit(problem)
    ans = client.chat_json('Reply only with JSON {"ok": true, "model": "<your name>"}', 'ping')
    print('Ollama OK, model {}: {}'.format(client.model, ans))


def cmd_review(args):
    from .config import AgentConfig
    from .llm import review_backtest
    cfg = AgentConfig.load(args.model)
    with open(os.path.join(args.model, 'backtest.json'), encoding='utf-8') as f:
        results = json.load(f)
    client = _llm_client(args)
    problem = client.check()
    if problem:
        sys.exit(problem)
    print(json.dumps(review_backtest(client, cfg, results), indent=2, ensure_ascii=False))


def cmd_live(args):
    from .live import LiveTrader
    from .train import load_model
    from . import presets
    from .gym_env import RLPolicy
    from .mt5_data import resolve_symbol
    model, cfg = load_model(args.model, device='cpu')
    pre = presets.get(cfg.preset)
    mt5 = _mt5(args, cfg.symbol)
    cfg.symbol = resolve_symbol(cfg.symbol)
    args.lot = args.lot or pre.get('lot', 0.01)
    if args.max_daily_loss is None and args.mode == 'demo':
        args.max_daily_loss = pre.get('max_daily_loss')
    confirm = False
    if args.mode == 'live':
        if not args.i_understand_live_risk:
            sys.exit('Handel live wymaga flagi --i-understand-live-risk (prawdziwe pieniadze!).')
        acc = mt5.account_info()
        typed = input('HANDEL NA PRAWDZIWYCH PIENIADZACH. Konto {} ({}), saldo {:.2f} {}. '
                      'Wpisz numer konta aby potwierdzic: '.format(acc.login, acc.server, acc.balance, acc.currency))
        confirm = typed.strip() == str(acc.login)
        if not confirm:
            sys.exit('Niepotwierdzone - koncze.')
    supervisor = None
    if args.llm_veto:
        from .llm import LLMSupervisor
        client = _llm_client(args)
        problem = client.check()
        if problem:
            sys.exit(problem)
        supervisor = LLMSupervisor(client, cfg)
    trader = LiveTrader(cfg, RLPolicy(model), mt5, mode=args.mode, dry_run=args.dry_run, lot=args.lot,
                        max_daily_loss=args.max_daily_loss, max_spread=args.max_spread, supervisor=supervisor,
                        poll_interval=args.poll, close_on_exit=args.close_on_exit, log_path=args.log,
                        confirm_live=confirm)
    if args.fake_mt5:  # symulator: czas plynie szybciej, zeby bylo widac decyzje
        import time
        from .fake_mt5 import _TF_SECONDS
        from .mt5_data import timeframe_const
        dt = cfg.step_len * 0.5 if cfg.source == 'ticks' else _TF_SECONDS[timeframe_const(cfg.timeframe)]
        trader.run(max_iterations=args.max_iterations, sleep=lambda s: (mt5.advance(dt), time.sleep(0.05)))
    else:
        trader.run(max_iterations=args.max_iterations)
    print('Koniec. Decyzji: {}. Log: {}'.format(trader.decisions, args.log))


def cmd_doctor(args):
    from .doctor import run
    from .hardware import detect
    model = args.llm_model or detect().llm_model
    sys.exit(run(model, args.ollama_host or 'http://localhost:11434', args.symbol, skip_mt5=args.skip_mt5))


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m tradebot', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-v', '--verbose', action='store_true')
    sub = p.add_subparsers(dest='cmd', required=True)

    def llm_args(sp):
        sp.add_argument('--llm-model', help='model Ollama (domyslnie dobrany do GPU, np. qwen3:14b)')
        sp.add_argument('--ollama-host', help='domyslnie http://localhost:11434')

    sub.add_parser('hw').set_defaults(func=cmd_hw)

    sp = sub.add_parser('fetch')
    sp.add_argument('--preset', help='np. xauusd (ustawia symbol, swiece M5, 2 lata historii)')
    sp.add_argument('--symbol')
    sp.add_argument('--source', choices=['ticks', 'bars'])
    sp.add_argument('--timeframe', help='M1, M5, H1... (domyslnie M1)')
    sp.add_argument('--start', help='np. 2024-01-01 (UTC)')
    sp.add_argument('--end', help='domyslnie teraz')
    sp.add_argument('--days', type=int, help='gdy brak --start (domyslnie 10)')
    sp.add_argument('--price-source', choices=['mid', 'last'], default='mid')
    sp.add_argument('--out', required=True, help='plik .csv albo .h5 (szybszy, mniejszy)')
    sp.add_argument('--fake-mt5', action='store_true')
    sp.set_defaults(func=cmd_fetch)

    sp = sub.add_parser('train')
    sp.add_argument('--data', required=True)
    sp.add_argument('--out', required=True, help='katalog modelu, np. models/eurusd')
    sp.add_argument('--symbol')
    sp.add_argument('--algo', choices=['PPO', 'DQN', 'ppo', 'dqn'], default='PPO')
    sp.add_argument('--preset', help='domyslnie z pliku .meta.json danych')
    sp.add_argument('--timesteps', type=int, help='domyslnie 1M (xauusd: 3M)')
    sp.add_argument('--obs-len', type=int, help='domyslnie 64 (xauusd: 48)')
    sp.add_argument('--step-len', type=int, help='domyslnie 16 (xauusd: 1)')
    sp.add_argument('--max-position', type=int, help='domyslnie 3')
    sp.add_argument('--fee', type=float, help='koszt zamkniecia 1 jednostki w cenie (domyslnie mediana spreadu)')
    sp.add_argument('--n-envs', type=int, help='domyslnie liczba watkow CPU - 2')
    sp.add_argument('--device', choices=['cuda', 'cpu'], help='domyslnie cuda jesli dostepna')
    sp.add_argument('--test-frac', type=float, default=0.2)
    sp.add_argument('--seed', type=int, default=0)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser('backtest')
    sp.add_argument('--model', required=True)
    sp.add_argument('--data', required=True)
    sp.add_argument('--policy', choices=['rl', 'random', 'llm'], default='rl')
    sp.add_argument('--last-days', type=int, help='tylko ostatnie N dni (przydatne dla --policy llm)')
    sp.add_argument('--out', help='zapisz transakcje do CSV')
    llm_args(sp)
    sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser('llm-check')
    llm_args(sp)
    sp.set_defaults(func=cmd_llm_check)

    sp = sub.add_parser('review')
    sp.add_argument('--model', required=True)
    llm_args(sp)
    sp.set_defaults(func=cmd_review)

    sp = sub.add_parser('live')
    sp.add_argument('--model', required=True)
    sp.add_argument('--mode', choices=['demo', 'live'], default='demo')
    sp.add_argument('--dry-run', action='store_true', help='bez zlecen, pozycje na papierze')
    sp.add_argument('--lot', type=float, help='wolumen 1 jednostki agenta (domyslnie 0.01)')
    sp.add_argument('--max-daily-loss', type=float, help='w walucie konta; wymagane dla live')
    sp.add_argument('--max-spread', type=float, help='nie otwieraj pozycji gdy spread (w cenie) wiekszy '
                    '(domyslnie z treningu: xauusd = 3 x mediana)')
    sp.add_argument('--llm-veto', action='store_true', help='LLM zatwierdza kazda akcje agenta')
    sp.add_argument('--poll', type=float, default=1.0, help='sekundy miedzy odczytami')
    sp.add_argument('--close-on-exit', action='store_true')
    sp.add_argument('--log', default='live_trades.csv')
    sp.add_argument('--max-iterations', type=int)
    sp.add_argument('--i-understand-live-risk', action='store_true')
    sp.add_argument('--fake-mt5', action='store_true')
    llm_args(sp)
    sp.set_defaults(func=cmd_live)

    sp = sub.add_parser('doctor')
    sp.add_argument('--symbol', default='XAUUSD')
    sp.add_argument('--skip-mt5', action='store_true')
    llm_args(sp)
    sp.set_defaults(func=cmd_doctor)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='[%(asctime)s] %(levelname)s %(message)s', force=True)
    args.func(args)


if __name__ == '__main__':
    main()
