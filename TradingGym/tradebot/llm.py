"""Warstwa LLM przez Ollama (lokalnie, http://localhost:11434).

LLM nie zastepuje uczenia RL - siec (PPO/DQN) uczy sie na danych. LLM pelni trzy role:
  * LLMPolicy      - samodzielny agent decydujacy na podstawie podsumowania rynku (do porownania z RL),
  * LLMSupervisor  - "drugi klucz" na zywo: zatwierdza albo wstrzymuje akcje agenta RL,
  * review_backtest- analizuje wyniki backtestu i proponuje zmiany parametrow treningu.

Model dobierany automatycznie do VRAM (hardware.py); dla RTX 5080 16 GB: qwen2.5:14b (~9 GB).
Instalacja: https://ollama.com -> `ollama pull <model>` (`python -m tradebot hw` pokaze ktory).
"""
import json
import logging
import urllib.error
import urllib.request

from .features import market_summary

log = logging.getLogger(__name__)

DEFAULT_MODEL = 'qwen2.5:14b'
DEFAULT_HOST = 'http://localhost:11434'
ACTIONS = {'hold': 0, 'buy': 1, 'sell': 2}

_RULES = ('Actions: "buy" = add 1 long unit or close 1 short unit; "sell" = add 1 short unit or close 1 long '
          'unit; "hold" = do nothing. Every closed unit pays a fee, so avoid overtrading. Returns and PnL are in '
          'basis points (bps). Never exceed max_position_units.')


class OllamaClient:
    def __init__(self, model=DEFAULT_MODEL, host=DEFAULT_HOST, timeout=60.0, options=None):
        self.model = model
        self.options = dict(options or {})
        self.host = host.rstrip('/')
        self.timeout = timeout

    def _request(self, path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.host + path, data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def models(self):
        return [m['name'] for m in self._request('/api/tags').get('models', [])]

    def check(self):
        """Zwraca None gdy OK, albo opis problemu."""
        try:
            names = self.models()
        except (urllib.error.URLError, OSError) as e:
            return 'Ollama nie odpowiada pod {} ({}). Uruchom aplikacje Ollama.'.format(self.host, e)
        if not any(n == self.model or n.split(':')[0] == self.model for n in names):
            return 'Brak modelu {} w Ollama. Uruchom: ollama pull {} (masz: {})'.format(
                self.model, self.model, ', '.join(names) or 'brak')
        return None

    def chat_json(self, system, user):
        out = self._request('/api/chat', {
            'model': self.model, 'stream': False, 'format': 'json',
            'options': dict(self.options, temperature=0),
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
        })
        return json.loads(out['message']['content'])


class LLMPolicy:
    """Agent oparty wylacznie na LLM. Wolny (1 zapytanie na decyzje) - do backtestu na malej probce."""
    name = 'llm'

    def __init__(self, client, cfg):
        self.client = client
        self.cfg = cfg
        self.last_reason = ''

    def act(self, obs, window, position, unrealized):
        summary = market_summary(window, self.cfg.features, position, self.cfg.max_position, unrealized)
        system = ('You are a disciplined intraday trading agent. ' + _RULES +
                  ' Reply only with JSON: {"action": "hold|buy|sell", "confidence": 0..1, "reason": "short"}')
        try:
            ans = self.client.chat_json(system, json.dumps({'symbol': self.cfg.symbol, 'market': summary}))
            self.last_reason = str(ans.get('reason', ''))
            return ACTIONS.get(str(ans.get('action', 'hold')).lower(), 0)
        except Exception as e:  # blad LLM = nie handluj
            log.warning('LLM blad, akcja hold: %s', e)
            self.last_reason = 'llm error'
            return 0


class LLMSupervisor:
    """Zatwierdza/wstrzymuje akcje agenta RL. Przy bledzie LLM akcja jest wstrzymywana (bezpieczniej)."""

    def __init__(self, client, cfg):
        self.client = client
        self.cfg = cfg

    def review(self, action, window, position, unrealized):
        if action == 0:
            return True, 'hold'
        summary = market_summary(window, self.cfg.features, position, self.cfg.max_position, unrealized)
        proposed = {1: 'buy', 2: 'sell'}[action]
        system = ('You are a risk manager supervising a reinforcement-learning trading agent. ' + _RULES +
                  ' Approve the proposed action unless it is clearly risky (e.g. extreme volatility, very wide '
                  'spread, adding to a large losing position against a strong trend). Reply only with JSON: '
                  '{"approve": true|false, "reason": "short"}')
        try:
            ans = self.client.chat_json(system, json.dumps({'symbol': self.cfg.symbol, 'proposed_action': proposed,
                                                            'market': summary}))
            return bool(ans.get('approve', False)), str(ans.get('reason', ''))
        except Exception as e:
            log.warning('LLM supervisor blad, wstrzymuje akcje: %s', e)
            return False, 'llm error: {}'.format(e)


def review_backtest(client, cfg, results):
    """results: lista podsumowan z run_backtest (np. RL vs random). Zwraca dict z ocena i sugestiami."""
    system = ('You are a quantitative researcher reviewing a reinforcement-learning trading agent trained on '
              'MetaTrader 5 data with PPO/DQN. Tunable parameters: obs_len (window length), step_len (decision '
              'interval), max_position, fee, algo (PPO|DQN), training timesteps, data range and features. '
              'Compare the RL agent with the random baseline, judge overfitting risk and whether it is fit for '
              'demo trading. Answer in Polish. Reply only with JSON: {"assessment": "...", '
              '"ready_for_demo": true|false, "suggestions": [{"param": "...", "value": "...", "why": "..."}]}')
    user = json.dumps({'config': cfg.__dict__, 'backtests': results}, default=str)
    return client.chat_json(system, user)
