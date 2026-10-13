"""Rownolegle srodowiska w paczkach: kazdy proces liczy kilka-kilkanascie kopii rynku naraz.

Standardowy SubprocVecEnv trzyma 1 srodowisko na proces. Krok TradingGym trwa ~0,1 ms, wiec prawie
caly czas zjada komunikacja miedzy procesami i inferencja sieci na 18 obserwacjach - CPU i GPU stoja.
Tu kazdy z `n_workers` procesow trzyma `envs_per_worker` srodowisk: jedna wymiana danych na paczke,
a siec liczy setki obserwacji naraz (dobre wykorzystanie GPU).
"""
import multiprocessing as mp

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv


def _worker(remote, parent_remote, fns_wrapper):
    parent_remote.close()
    envs = [fn() for fn in fns_wrapper.var]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                obs, rews, dones, infos = [], [], [], []
                for env, action in zip(envs, data):
                    o, r, term, trunc, info = env.step(action)
                    done = term or trunc
                    info['TimeLimit.truncated'] = trunc and not term
                    if done:
                        info['terminal_observation'] = o
                        o, _ = env.reset()
                    obs.append(o)
                    rews.append(r)
                    dones.append(done)
                    infos.append(info)
                remote.send((np.stack(obs), np.asarray(rews, dtype=np.float32), np.asarray(dones), infos))
            elif cmd == 'reset':
                remote.send(np.stack([env.reset(seed=s)[0] for env, s in zip(envs, data)]))
            elif cmd == 'get_spaces':
                remote.send((envs[0].observation_space, envs[0].action_space))
            elif cmd == 'get_attr':
                remote.send([getattr(e, data) for e in envs])
            elif cmd == 'set_attr':
                for e in envs:
                    setattr(e, data[0], data[1])
                remote.send(None)
            elif cmd == 'env_method':
                name, args, kwargs = data
                remote.send([getattr(e, name)(*args, **kwargs) for e in envs])
            elif cmd == 'close':
                remote.close()
                break
    except (EOFError, KeyboardInterrupt):
        pass


class BatchedSubprocVecEnv(VecEnv):
    def __init__(self, env_fns, envs_per_worker):
        self.waiting, self.closed = False, False
        groups = [env_fns[i:i + envs_per_worker] for i in range(0, len(env_fns), envs_per_worker)]
        self.sizes = [len(g) for g in groups]
        ctx = mp.get_context('spawn')
        self.remotes, work_remotes = zip(*[ctx.Pipe() for _ in groups])
        self.processes = []
        for work, remote, group in zip(work_remotes, self.remotes, groups):
            p = ctx.Process(target=_worker, args=(work, remote, CloudpickleWrapper(group)), daemon=True)
            p.start()
            self.processes.append(p)
            work.close()
        self.remotes[0].send(('get_spaces', None))
        obs_space, act_space = self.remotes[0].recv()
        super().__init__(len(env_fns), obs_space, act_space)

    def _split(self, items):
        out, i = [], 0
        for n in self.sizes:
            out.append(items[i:i + n])
            i += n
        return out

    def step_async(self, actions):
        for remote, acts in zip(self.remotes, self._split(list(actions))):
            remote.send(('step', acts))
        self.waiting = True

    def step_wait(self):
        results = [r.recv() for r in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.concatenate(obs), np.concatenate(rews), np.concatenate(dones), [i for part in infos for i in part]

    def reset(self):
        for remote, seeds in zip(self.remotes, self._split(list(self._seeds))):
            remote.send(('reset', seeds))
        obs = np.concatenate([r.recv() for r in self.remotes])
        self._reset_seeds()
        self._reset_options()
        return obs

    def close(self):
        """Odporne na Ctrl+C: na Windows przerwanie trafia tez do procesow-pracownikow, ktore konca sie same -
        wtedy polaczenie jest juz zerwane i tylko sprzatamy."""
        if self.closed:
            return
        for r in self.remotes:
            try:
                if self.waiting:
                    r.recv()
                r.send(('close', None))
            except (EOFError, BrokenPipeError, ConnectionResetError, OSError):
                pass
        for p in self.processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        self.closed = True

    def _gather(self, cmd, data, indices):
        for r in self.remotes:
            r.send((cmd, data))
        flat = [x for r in self.remotes for x in (r.recv() or [None] * 0)]
        idx = self._get_indices(indices)
        return [flat[i] for i in idx] if flat else []

    def get_attr(self, attr_name, indices=None):
        return self._gather('get_attr', attr_name, indices)

    def set_attr(self, attr_name, value, indices=None):
        for r in self.remotes:
            r.send(('set_attr', (attr_name, value)))
        for r in self.remotes:
            r.recv()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self._gather('env_method', (method_name, method_args, method_kwargs), indices)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))
