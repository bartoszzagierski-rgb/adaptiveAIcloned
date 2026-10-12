# tradebot: MT5 + trening RL + LLM (Ollama) + handel demo i live

Moduł `tradebot/` korzysta z TradingGym jako symulatora i dodaje cztery rzeczy:

1. **MetaTrader 5.** Pobiera historyczne ticki albo świece do treningu, a w trakcie handlu odczytuje bieżące ticki.
2. **Trening RL.** Agent uczy się algorytmem PPO albo DQN (biblioteka stable-baselines3). Trening wykorzystuje cały sprzęt: wszystkie rdzenie CPU, GPU, RAM i dysk SSD.
3. **LLM przez Ollamę.** Model jest dobierany automatycznie do karty graficznej. Na RTX 5080 (16 GB) to `qwen3:14b`.
4. **Handel.** Możliwe tryby: demo, live oraz dry-run (decyzje bez wysyłania zleceń). Wszystkie mają zabezpieczenia.

> **Uwaga.** To narzędzie badawcze. Agent RL może dawać dobre wyniki na danych historycznych, a mimo to tracić pieniądze na żywo. Zawsze idź po kolei: backtest → dry-run → demo przez kilka tygodni → live z małym lotem i limitem straty.

## Jak to działa razem

```
MT5 (ticki/świece) ──fetch──► data/*.h5 ──train──► models/<nazwa>/model.zip + config.json + backtest.json
                                                     │
                     LLM (Ollama) ◄──review──────────┤  ocena wyników i propozycje parametrów
                                                     ▼
MT5 (bieżące ticki) ──► agent RL ──► [LLM zatwierdza/wstrzymuje] ──► zlecenia MT5 (demo/live)
```

**Rola LLM.** LLM sam się nie uczy. Uczy się sieć RL, trenowana na tysiącach dni symulacji. LLM ma trzy zadania:
- `review`: czyta wyniki backtestu i proponuje zmiany parametrów treningu.
- `--llm-veto`: w handlu działa jak drugi klucz. Może wstrzymać akcję agenta, którą uzna za ryzykowną. Gdy LLM nie odpowiada, akcja jest wstrzymywana.
- `--policy llm`: LLM jako samodzielny agent, do porównania z RL. Jest wolny i nadaje się tylko na kilka dni danych.

## Wykorzystanie sprzętu T7 (Core Ultra 7, RTX 5080 16 GB, 64 GB RAM, SSD 2 TB)

Polecenie `python -m tradebot hw` pokazuje wykryty sprzęt i dobrane ustawienia:

| Zasób | Jak jest używany |
|---|---|
| CPU | `n_envs` = liczba wątków − 2. Każda kopia środowiska działa w osobnym procesie (SubprocVecEnv). Dwa wątki zostają dla systemu i MT5. |
| GPU | Sieć PPO/DQN liczy się na CUDA: większa sieć `[512, 512, 256]` i batch 4096. Ollama trzyma cały model LLM w VRAM (`num_gpu=999`). |
| RAM | Dane są trzymane w pamięci. Replay buffer DQN zajmuje do ok. 25% RAM, czyli przy 64 GB miliony przejść. |
| SSD | Dane w formacie `.h5` oraz checkpoint modelu co 200 tys. kroków w `models/<nazwa>/checkpoints/`. |

Dobór modelu LLM do ilości VRAM (model mieści się w całości na GPU obok sieci RL):

| VRAM | Model |
|---|---|
| ≥ 22 GB | qwen3:32b |
| ≥ 14 GB | **qwen3:14b (RTX 5080)** |
| ≥ 8 GB | qwen3:8b |

Model można zmienić parametrem `--llm-model`. Przy 64 GB RAM da się uruchomić `qwen3:32b` z częściowym przeniesieniem do RAM. Będzie jednak kilka razy wolniejszy, a przy handlu na żywo liczy się czas odpowiedzi.

Sieć RL jest mała (MLP), więc główne obciążenie przy treningu przypada na CPU (symulacja), a GPU jest wykorzystane tylko częściowo. To normalne.

## Instalacja (Windows, PowerShell, w folderze `TradingGym`)

```powershell
python -m pip install -r requirements.txt
# PyTorch z CUDA dla RTX 5080 (seria 50 wymaga CUDA 12.8 lub nowszej):
python -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu130
#   (jeśli ten adres nie zadziała, wybierz polecenie na https://pytorch.org/get-started/locally/ : Windows, Pip, CUDA 12.8+)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m tradebot hw
```

Ollama: zainstaluj z https://ollama.com, a potem:
```powershell
ollama pull qwen3:14b
python -m tradebot llm-check
```

MT5: uruchom terminal MT5 i zaloguj się na konto **demo**. Pakiet `MetaTrader5` łączy się z otwartym terminalem. Opcjonalnie dane logowania można podać w zmiennych środowiskowych (nie zapisuj ich w plikach):
```powershell
$env:MT5_LOGIN="12345678"; $env:MT5_PASSWORD="..."; $env:MT5_SERVER="Broker-Demo"
```

## Użycie krok po kroku

```powershell
# 1. Dane: 30 dni ticków EURUSD (albo --source bars --timeframe M1)
python -m tradebot fetch --symbol EURUSD --days 30 --out data/eurusd.h5

# 2. Trening (wszystkie rdzenie + GPU), na koniec backtest na ostatnich 20% dni
python -m tradebot train --data data/eurusd.h5 --out models/eurusd --timesteps 2000000

# 3. Ocena przez LLM
python -m tradebot review --model models/eurusd

# 4. Porównanie na danych
python -m tradebot backtest --model models/eurusd --data data/eurusd.h5 --policy rl
python -m tradebot backtest --model models/eurusd --data data/eurusd.h5 --policy random

# 5. Dry-run na żywych tickach (bez zleceń)
python -m tradebot live --model models/eurusd --mode demo --dry-run --lot 0.01

# 6. Konto demo, z nadzorem LLM i limitami
python -m tradebot live --model models/eurusd --mode demo --lot 0.01 --max-daily-loss 50 --max-spread 0.0002 --llm-veto

# 7. Konto realne: dopiero po tygodniach na demo
python -m tradebot live --model models/eurusd --mode live --lot 0.01 --max-daily-loss 20 --i-understand-live-risk
```

Cały przebieg bez MT5 (symulowane ceny, tylko do sprawdzenia, czy wszystko działa):
```powershell
python -m tradebot fetch --fake-mt5 --symbol EURUSD --source bars --days 5 --out data/fake.csv
python -m tradebot train --data data/fake.csv --out models/fake --timesteps 20000
python -m tradebot live --fake-mt5 --model models/fake --dry-run --max-iterations 200 --poll 0
```

## Sprawdzenie instalacji

```bat
python -m tradebot doctor
```
Polecenie sprawdza: wersję Pythona i pakietów, sterownik NVIDIA (dla RTX 50 wymagany 570 lub nowszy), PyTorch z CUDA i obsługą Twojej karty, program i serwer Ollama (wersja 0.5.0 lub nowsza), model (odpowiedź w JSON, ile modelu siedzi w VRAM), MT5 (konto, Algo Trading, limit świec, symbol XAUUSD u brokera).

## XAUUSD (złoto): `--preset xauusd`

```bat
python -m tradebot fetch --preset xauusd --out data/xauusd_m5.h5
python -m tradebot train --data data/xauusd_m5.h5 --out models/xauusd
python -m tradebot review --model models/xauusd
python -m tradebot live --model models/xauusd --mode demo --dry-run
python -m tradebot live --model models/xauusd --mode demo --llm-veto
```

Co ustawia preset (train i live odczytują go same z danych i modelu):

| Ustawienie | Wartość | Dlaczego |
|---|---|---|
| dane | świece **M5**, **10 lat** (albo tyle, ile ma broker) | ticków złota jest setki tysięcy dziennie i są zaszumione; 10 lat obejmuje różne okresy: spokojne lata, COVID, wojnę, rekordy cen |
| cechy | cena, spread, wolumen, zakres świecy | zakres świecy (high−low) dobrze pokazuje zmienność złota |
| okno / decyzja | 48 świec (4 h) / co świecę (5 min) | |
| koszt `fee_bps` | 1,5 × mediana spreadu z ostatniego roku, **jako % ceny** | spread ze świec MT5 to minimum, a dochodzi poślizg; % ceny, bo złoto podrożało przez 10 lat ok. 3,5 raza (1200 → 4200 USD) i stały koszt w USD byłby zawyżony dla starych lat |
| `max_spread_bps` | 3 × mediana spreadu (jako % ceny) | brak nowych pozycji przy newsach i rolloverze |
| sesja | handel od **05:00**, zamknięcie pozycji po **23:30** czasu serwera | tak samo jak w treningu: dzień = epizod, pierwsze 4 h to tylko obserwacja, pozycja zamknięta przed przerwą (bez swapów) |
| lot / limity | 0,01 lota na jednostkę, max 3 jednostki, 2% dziennie, 10% łącznie | zobacz „Ryzyko” niżej |
| trening | **do skutku**: sprawdzenie co 2 mln kroków, koniec po 10 sprawdzeniach bez poprawy (bezpieczniki: 300 mln kroków / 72 h) | zobacz „Trening do skutku i egzamin” niżej |

Symbol jest wyszukiwany automatycznie, więc zadziała też u brokerów z nazwami `XAUUSDm`, `XAUUSD.pro` czy `GOLD`.

**Ważne przed `fetch`:** w MT5 ustaw *Narzędzia → Opcje → Wykresy → Maks. liczba świec na wykresie* na **Unlimited**. Inaczej MT5 odda tylko ok. 100 tys. świec, czyli mniej niż rok M5. `doctor` to sprawdza. 10 lat M5 to ok. 700 tys. świec. Brokerzy CFD często nie mają aż tak głębokiej historii M5: `fetch` pobierze wtedy maksimum dostępne i wypisze ostrzeżenie, od kiedy zaczynają się dane. Jeśli przy pierwszym pobraniu przyjdzie mniej danych, otwórz w MT5 wykres XAUUSD M5, przewiń go maksymalnie w lewo (klawisz Home), żeby terminal dociągnął historię z serwera, i uruchom `fetch` ponownie.

## Aktualizacja kodu bez pobierania ZIP: GitHub Desktop

1. Zainstaluj GitHub Desktop z https://desktop.github.com i zaloguj się na swoje konto GitHub (*File → Options → Accounts*).
2. *File → Clone repository → GitHub.com* → wybierz `bartoszzagierski-rgb/adaptiveAIcloned`, *Local path* np. `C:\Users\bzagi\adaptiveAIcloned` → **Clone**.
3. U góry kliknij **Current branch** i wybierz `claude/keen-bell-qfmyy8`. Po scaleniu PR wybierz `main`.
4. Gdy pojawią się zmiany, kliknij **Fetch origin**, a potem **Pull origin**.
5. Uruchamiaj z folderu `C:\Users\bzagi\adaptiveAIcloned\TradingGym`. Dane (`data/`), modele (`models/`) i logi są w `.gitignore`, więc aktualizacja ich nie nadpisze.

## Trening do skutku i egzamin

Dane dzielone są chronologicznie na trzy części:

```
|------------------- trening (~9 lat) -------------------|-- walidacja 6 mies. --|-- EGZAMIN 6 mies. --|
```

- **Trening.** Agent uczy się tylko na tej części.
- **Walidacja.** Co 2 mln kroków agent gra na tych 6 miesiącach (bez uczenia się na nich). Jeśli wynik się poprawił (Sharpe dziennych wyników, czyli zysk względem wahań), model jest zapisywany jako najlepszy. Gdy 10 sprawdzeń z rzędu nie przynosi poprawy, model przestał się uczyć i trening się kończy. Model, który przestaje handlować, nie liczy się jako poprawa. Postęp na bieżąco trafia do `models\xauusd\progress.json`.
- **Egzamin.** Ostatnie 6 miesięcy, których model nie widział ani w treningu, ani w walidacji. Na samym końcu gra na nich najlepszy model i agent losowy. Werdykt:
  - **ZDAŁ**: zarabia na nowych danych, Sharpe ≥ 0,5, lepszy od losowego, handluje. Następny krok: dry-run i demo.
  - **SŁABO**: lepszy od losowego, ale nie zarabia stabilnie. Nie nadaje się do handlu.
  - **NIE ZDAŁ**: nie lepszy od losowego albo prawie nie handluje.

Wynik zapisuje się w `models\xauusd\backtest.json`: `exam`, `rl_test` (egzamin), `random_test`, `rl_val`, `rl_train_last_6m`. Jeśli wynik na treningu jest dużo lepszy niż na egzaminie, model nauczył się historii na pamięć (przeuczenie).

Trening można przerwać w dowolnej chwili przez **Ctrl+C**. Egzamin przejdzie wtedy od razu z najlepszym dotąd modelem. Limity zmienisz parametrami `--patience`, `--eval-every`, `--max-hours`, `--timesteps` i `--test-months`.

## Uczenie ciągłe: douczanie co tydzień (`retrain`)

Agent **nie uczy się w trakcie handlu**. Na żywo działa na zamrożonym modelu, a uczy się osobno, na nowych danych, metodą champion/challenger:

```bat
python -m tradebot retrain --model models\xauusd
```
1. Pobiera z MT5 ostatnie 180 dni (świece M5).
2. Ostatnie 10 dni odkłada jako sprawdzian, którego model nie widzi.
3. Douczanie kopii obecnego modelu (challenger) na pozostałych dniach, 500 tys. kroków.
4. Obecny model, challenger i agent losowy grają na dniach sprawdzianu.
5. Challenger zastępuje model **tylko jeśli** wypadł lepiej od obecnego i od losowego. Stary model trafia do `models\xauusd\history\`, a każdy przebieg zapisuje się w `retrain_log.json`.
6. Uruchomiony `live` sam przeładuje nowy model na początku kolejnego dnia. Otwarte pozycje zostają.

Dlaczego nie uczenie się w locie, transakcja po transakcji:
- Z rynku przychodzi ok. 200 decyzji dziennie, a trening potrzebuje milionów. Uczenie na kilkuset przykładach przestawia sieć pod ostatnie godziny szumu.
- Nauka wymaga eksploracji, czyli celowo przypadkowych decyzji. Na żywo oznacza to losowe zlecenia za prawdziwe pieniądze.
- Bez sprawdzianu nie wiadomo, czy model się poprawił, czy zepsuł. Champion/challenger tego pilnuje.

Automatycznie co sobotę o 10:00, kiedy rynek złota jest zamknięty, a MT5 włączony. W cmd **jako administrator**, jedna linia:
```bat
schtasks /Create /SC WEEKLY /D SAT /ST 10:00 /TN "tradebot retrain" /TR "cmd /c cd /d C:\Users\bzagi\Documents\GitHub\adaptiveAIcloned\TradingGym && C:\Users\bzagi\AppData\Local\Python\pythoncore-3.12-64\python.exe -m tradebot retrain --model models\xauusd >> retrain.log 2>&1"
```
Sprawdzenie: `schtasks /Query /TN "tradebot retrain"`. Usunięcie: `schtasks /Delete /TN "tradebot retrain" /F`.

## Ryzyko: lot 0,01 i konto 10 000 zł

Bot zawsze handluje **0,01 lota** na jednostkę (`--lot`, domyślnie 0,01) i ma najwyżej 3 jednostki, czyli maksymalnie 0,03 lota. Saldo konta odczytuje sam z MT5 i od niego liczy limity:

| Zabezpieczenie | Domyślnie (xauusd) | Przy 10 000 zł |
|---|---|---|
| `--max-daily-loss-pct` | 2% kapitału na dzień | stop po stracie 200 zł w ciągu dnia: zamyka pozycje i kończy |
| `--max-drawdown-pct` | 10% od uruchomienia | stop po łącznej stracie 1 000 zł |
| kontrola depozytu | depozyt na nową pozycję ≤ 50% wolnego depozytu | przy dźwigni 1:20 jedna jednostka (1 uncja przy ok. 4170 USD) to ok. 210 USD, czyli ok. 750–800 zł depozytu |
| `--risk-capital` | brak | jeśli konto jest większe (albo w innej walucie), limity liczą się od tej kwoty |

Skala ruchów: 0,01 lota złota to 1 uncja, więc ruch ceny o 1 USD to 1 USD zysku lub straty na jednostkę (ok. 3,6–4 zł). Typowy dzienny zakres złota to kilkadziesiąt USD, czyli przy 3 jednostkach kilkaset zł. Dlatego limit dzienny 2% jest ważny.

Limity liczą się w **walucie konta**. Jeśli konto demo jest w USD, a chcesz limitów jak dla 10 000 zł, podaj `--risk-capital` w USD, np. `--risk-capital 2700`.

## Zabezpieczenia w handlu

- `--mode demo` odmówi pracy na koncie realnym, a `--mode live` na koncie demo.
- `live` wymaga flagi `--i-understand-live-risk`, wpisania numeru konta i ustawienia `--max-daily-loss`.
- Limit dziennej straty: po przekroczeniu bot zamyka swoje pozycje i kończy pracę.
- `--max-spread`: przy szerokim spreadzie bot nie otwiera nowych pozycji.
- Bot rusza tylko własne pozycje, rozpoznawane po numerze magic 770077. Twoje ręczne pozycje zostają nietknięte.
- Maksymalna pozycja jest wzięta z treningu. Jedna jednostka agenta to `--lot` lotów.
- Utworzenie pliku `STOP` w katalogu roboczym zatrzymuje bota. Ctrl+C też kończy pracę bezpiecznie, a z `--close-on-exit` dodatkowo zamyka pozycje.
- W terminalu musi być włączony przycisk **Algo Trading**.
- Każda decyzja trafia do `live_trades.csv`.

## Parametry agenta

| Parametr | Znaczenie |
|---|---|
| `--obs-len` | ile ostatnich ticków lub świec widzi agent (domyślnie 64) |
| `--step-len` | co ile ticków lub świec agent podejmuje decyzję (16) |
| `--max-position` | maksymalna liczba jednostek long lub short (3) |
| `--fee` | koszt zamknięcia jednej jednostki w jednostkach ceny (domyślnie mediana spreadu) |
| `--algo` | PPO (domyślnie) albo DQN |

Agent widzi ceny jako odchylenie od ostatniej ceny, spread i wolumen (znormalizowane), a także swoją pozycję i niezrealizowany wynik. Nagroda to wynik mark-to-market w punktach bazowych, czyli zrealizowany zysk plus zmiana niezrealizowanego, po opłatach. Pozycja jest zamykana na koniec każdego dnia, tak jak w TradingGym.

## Testy

```powershell
python -m pytest tests -q
```
Testy używają symulatora MT5 i fałszywego LLM, więc nie wymagają terminala MT5 ani Ollamy.
