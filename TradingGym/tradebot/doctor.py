"""python -m tradebot doctor - sprawdza wersje i konfiguracje: Python, pakiety, GPU/CUDA, Ollama, model, MT5."""
import importlib.metadata as md
import json
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request

MIN_OLLAMA = (0, 5, 0)      # format JSON / structured outputs
MIN_DRIVER_RTX50 = 570      # sterownik NVIDIA wymagany przez RTX serii 50

_results = []


def _report(status, name, detail=''):
    _results.append(status)
    print('[{:4}] {:<28} {}'.format(status, name, detail))


def _ver(s):
    return tuple(int(x) for x in re.findall(r'\d+', s)[:3])


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ''


def check_python():
    v = sys.version_info
    _report('OK' if v >= (3, 10) else 'FAIL', 'Python', '{}.{}.{} ({})'.format(v.major, v.minor, v.micro, sys.executable))


def check_packages():
    required = ['pandas', 'numpy', 'matplotlib', 'colour', 'tables', 'gymnasium', 'stable-baselines3', 'torch']
    if platform.system() == 'Windows':
        required.append('MetaTrader5')
    for pkg in required:
        try:
            _report('OK', pkg, md.version(pkg))
        except md.PackageNotFoundError:
            _report('FAIL', pkg, 'brak -> python -m pip install -r requirements.txt')


def check_gpu():
    smi = _run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'])
    if smi:
        name, driver, mem = [x.strip() for x in smi.splitlines()[0].split(',')]
        rtx50 = bool(re.search(r'RTX\s*50\d0', name))
        ok = not rtx50 or _ver(driver)[0] >= MIN_DRIVER_RTX50
        _report('OK' if ok else 'FAIL', 'GPU / sterownik', '{}, sterownik {}, {}{}'.format(
            name, driver, mem, '' if ok else ' -> zaktualizuj sterownik do >= {}'.format(MIN_DRIVER_RTX50)))
    else:
        _report('WARN', 'GPU / sterownik', 'nvidia-smi niedostepne - brak GPU NVIDIA albo sterownika')
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            arch = 'sm_{}{}'.format(*cap)
            ok = arch in torch.cuda.get_arch_list()
            _report('OK' if ok else 'FAIL', 'PyTorch CUDA', 'torch {} CUDA {} {} ({}){}'.format(
                torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), arch,
                '' if ok else ' -> ta wersja torch nie wspiera tej karty, zainstaluj nowsza z CUDA 12.8+'))
        else:
            _report('WARN' if smi else 'OK', 'PyTorch CUDA', 'torch {} bez CUDA - trening na CPU{}'.format(
                torch.__version__, ' -> zainstaluj torch z CUDA (TRADEBOT.md)' if smi else ''))
    except ImportError:
        pass


def check_ollama(model, host):
    exe = shutil.which('ollama')
    if exe:
        out = _run([exe, '--version'])
        _report('OK', 'Ollama (program)', '{} | {}'.format(exe, out.splitlines()[-1] if out else '?'))
    else:
        _report('FAIL', 'Ollama (program)', 'nie znaleziono -> https://ollama.com/download')
    try:
        with urllib.request.urlopen(host + '/api/version', timeout=5) as r:
            ver = json.loads(r.read().decode())['version']
        ok = _ver(ver) >= MIN_OLLAMA
        _report('OK' if ok else 'FAIL', 'Ollama (serwer)', 'wersja {} pod {}{}'.format(
            ver, host, '' if ok else ' -> zaktualizuj do >= {}.{}.{}'.format(*MIN_OLLAMA)))
    except Exception as e:
        _report('FAIL', 'Ollama (serwer)', 'nie odpowiada pod {} ({}) -> uruchom aplikacje Ollama'.format(host, e))
        return
    from .llm import OllamaClient
    client = OllamaClient(model=model, host=host, timeout=180)
    problem = client.check()
    if problem:
        _report('FAIL', 'Model ' + model, problem)
        return
    t0 = time.time()
    try:
        ans = client.chat_json('Reply only with JSON {"ok": true}', 'ping')
        _report('OK' if ans.get('ok') else 'WARN', 'Model ' + model, 'odpowiedz JSON w {:.1f} s: {}'.format(
            time.time() - t0, ans))
    except Exception as e:
        _report('FAIL', 'Model ' + model, 'blad zapytania: {}'.format(e))
        return
    try:
        with urllib.request.urlopen(host + '/api/ps', timeout=5) as r:
            running = json.loads(r.read().decode()).get('models', [])
        m = next((m for m in running if m['name'].split(':')[0] == model.split(':')[0]), None)
        if m and m.get('size'):
            gpu = m.get('size_vram', 0) / m['size']
            _report('OK' if gpu > 0.99 else 'WARN', 'Model na GPU', '{:.0%} w VRAM ({:.1f} GB){}'.format(
                gpu, m['size'] / 1024 ** 3, '' if gpu > 0.99 else ' -> czesc na CPU, wolniej; wybierz mniejszy model'))
    except Exception:
        pass


def check_mt5(symbol):
    if platform.system() != 'Windows':
        _report('WARN', 'MetaTrader 5', 'pakiet MetaTrader5 dziala tylko na Windows')
        return
    from . import mt5_data
    try:
        mt5 = mt5_data.connect()
    except Exception as e:
        _report('FAIL', 'MetaTrader 5', str(e))
        return
    acc, term = mt5.account_info(), mt5.terminal_info()
    mode = {0: 'DEMO', 1: 'CONTEST', 2: 'REAL'}.get(acc.trade_mode, acc.trade_mode)
    _report('OK', 'MetaTrader 5', 'konto {} {} ({}), build {}'.format(acc.login, mode, acc.server,
                                                                      getattr(term, 'build', '?')))
    _report('OK' if term.trade_allowed else 'WARN', 'Algo Trading',
            'wlaczony' if term.trade_allowed else 'wylaczony -> wlacz przycisk "Algo Trading" w MT5')
    if getattr(term, 'maxbars', 0) and term.maxbars < 250_000:
        _report('WARN', 'Maks. swiec na wykresie', '{} -> ustaw Unlimited (Narzedzia > Opcje > Wykresy) dla '
                '2 lat M5'.format(term.maxbars))
    try:
        name = mt5_data.resolve_symbol(symbol)
        info, tick = mt5.symbol_info(name), mt5.symbol_info_tick(name)
        _report('OK', 'Symbol ' + symbol, '{} cena {} spread {:.2f} ({} pkt), lot min {}'.format(
            name, tick.bid, tick.ask - tick.bid, round((tick.ask - tick.bid) / info.point), info.volume_min))
    except Exception as e:
        _report('FAIL', 'Symbol ' + symbol, str(e))


def run(model, host, symbol, skip_mt5=False):
    print('=== tradebot doctor ===')
    check_python()
    check_packages()
    check_gpu()
    check_ollama(model, host)
    if not skip_mt5:
        check_mt5(symbol)
    fails = _results.count('FAIL')
    print('\nWynik: {} OK, {} ostrzezen, {} bledow'.format(_results.count('OK'), _results.count('WARN'), fails))
    return 1 if fails else 0
