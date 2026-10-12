"""Konfiguracja agenta zapisywana razem z modelem (models/<nazwa>/config.json).

Te same parametry musza byc uzyte przy trenowaniu, backteście i handlu live,
inaczej obserwacje beda mialy inny ksztalt/znaczenie niz te, na ktorych model sie uczyl.
"""
import json
import os
from dataclasses import asdict, dataclass, field

DEFAULT_FEATURES = ['price', 'spread', 'volume']


@dataclass
class AgentConfig:
    symbol: str = ''
    source: str = 'ticks'          # 'ticks' albo 'bars'
    timeframe: str = 'M1'          # tylko dla source='bars'
    price_source: str = 'mid'      # 'mid' = (bid+ask)/2, 'last' = cena transakcji (gielda)
    features: list = field(default_factory=lambda: list(DEFAULT_FEATURES))
    obs_len: int = 64              # ile tickow/swiec widzi agent
    step_len: int = 16             # co ile tickow/swiec agent podejmuje decyzje
    max_position: int = 3          # max liczba jednostek (1 jednostka = --lot lotow na zywo)
    fee: float = 0.0               # koszt zamkniecia 1 jednostki w jednostkach ceny (spread + prowizja)
    algo: str = 'PPO'

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, directory):
        with open(os.path.join(directory, 'config.json'), encoding='utf-8') as f:
            return cls(**json.load(f))
