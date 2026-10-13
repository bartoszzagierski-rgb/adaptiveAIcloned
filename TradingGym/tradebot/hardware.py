"""Wykrywanie sprzetu i dobor parametrow (CPU / GPU / RAM / SSD) oraz modelu LLM.

Dla Lenovo T7 (Core Ultra 7, RTX 5080 16 GB, 64 GB RAM):
  * CPU  -> wiele rownoleglych srodowisk (SubprocVecEnv), po jednym na rdzen (minus 2 dla systemu i MT5),
  * GPU  -> siec PPO/DQN na CUDA, wieksza siec i wieksze batch'e; Ollama trzyma caly model w VRAM,
  * RAM  -> duzy replay buffer dla DQN (do ~25% RAM), dane trzymane w pamieci,
  * SSD  -> checkpointy modelu co N krokow + zapis danych do .h5.
"""
import ctypes
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass

# (min VRAM GB, model) - model musi sie zmiescic w calosci w VRAM obok sieci RL (~1-2 GB)
LLM_BY_VRAM = [
    (22, 'qwen3:32b'),
    (14, 'qwen3:14b'),
    (8, 'qwen3:8b'),
    (0, 'qwen3:4b'),
]


@dataclass
class HardwareProfile:
    cpu_threads: int
    ram_gb: float
    gpu_name: str
    vram_gb: float
    cuda: bool
    free_disk_gb: float

    # --- dobrane ustawienia ---
    @property
    def n_envs(self):
        return max(1, min(32, self.cpu_threads - 2))

    @property
    def envs_per_worker(self):
        """Ile kopii rynku liczy jeden proces naraz (paczka dla sieci = n_envs x envs_per_worker)."""
        return 16 if self.cuda else 8

    @property
    def device(self):
        return 'cuda' if self.cuda else 'cpu'

    @property
    def net_arch(self):
        return [512, 512, 256] if self.cuda else [256, 256]

    @property
    def batch_size(self):
        return 4096 if self.cuda else 512

    def dqn_buffer_size(self, obs_dim):
        bytes_per_transition = obs_dim * 4 * 2 + 64  # obs + next_obs float32
        budget = self.ram_gb * 0.25 * 1024 ** 3
        return int(max(50_000, min(5_000_000, budget // bytes_per_transition)))

    @property
    def llm_model(self):
        if self.vram_gb == 0:  # brak GPU: model w RAM, wolniejszy -> maly
            return 'qwen3:14b' if self.ram_gb >= 32 else 'qwen3:8b' if self.ram_gb >= 16 else 'qwen3:4b'
        return next(m for v, m in LLM_BY_VRAM if self.vram_gb >= v)

    def ollama_options(self):
        return {'num_gpu': 999 if self.vram_gb else 0,  # wszystkie warstwy na GPU
                'num_thread': max(1, self.cpu_threads // 2),
                'num_ctx': 4096}

    def describe(self):
        d = asdict(self)
        d.update(n_envs=self.n_envs, envs_per_worker=self.envs_per_worker, device=self.device, net_arch=self.net_arch, batch_size=self.batch_size,
                 llm_model=self.llm_model)
        return d


def _ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / 1024 ** 3
    except ImportError:
        pass
    if platform.system() == 'Windows':
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('sullAvailExtendedVirtual', ctypes.c_ulonglong)]
        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(st)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ullTotalPhys / 1024 ** 3
    try:
        with open('/proc/meminfo') as f:
            return int(f.readline().split()[1]) / 1024 ** 2
    except OSError:
        return 8.0


def _gpu():
    """(nazwa, VRAM GB, czy torch widzi CUDA)."""
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return p.name, p.total_memory / 1024 ** 3, True
    except Exception:
        pass
    if shutil.which('nvidia-smi'):  # GPU jest, ale torch bez CUDA -> LLM nadal moze go uzyc
        try:
            out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
            name, mem = [x.strip() for x in out.split(',')]
            return name, float(mem) / 1024, False
        except Exception:
            pass
    return 'brak', 0.0, False


def detect(path='.'):
    path = os.path.abspath(path)
    while not os.path.exists(path):  # katalog modelu moze jeszcze nie istniec
        path = os.path.dirname(path)
    name, vram, cuda = _gpu()
    return HardwareProfile(cpu_threads=os.cpu_count() or 1, ram_gb=round(_ram_gb(), 1), gpu_name=name,
                           vram_gb=round(vram, 1), cuda=cuda,
                           free_disk_gb=round(shutil.disk_usage(path).free / 1024 ** 3, 1))


def configure_torch(profile):
    try:
        import torch
        torch.set_num_threads(max(1, profile.cpu_threads // 2))
        if profile.cuda:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
    except ImportError:
        pass
