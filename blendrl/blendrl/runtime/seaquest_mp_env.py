"""Multiprocessing Seaquest vector environment.

The original vector environment steps every Atari instance serially in the
training process. This backend preserves its wrappers and state representation
while distributing the environment steps across worker processes.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from hackatari.core import HackAtari
from ocatari.ram.seaquest import MAX_NB_OBJECTS

from blendrl.env_utils import make_env
from blendrl.env_vectorized import VectorizedNudgeBaseEnv


REPO_ROOT = Path(__file__).resolve().parents[2]
REWARD_PATH = REPO_ROOT / "in" / "envs" / "seaquest" / "blenderl_reward.py"


def _default_start_method() -> str:
    return "forkserver" if sys.platform != "win32" else "spawn"


def _resolve_num_workers(requested_workers: int, n_envs: int) -> int:
    if requested_workers > 0:
        return min(requested_workers, n_envs)
    return max(1, min(n_envs, 16, os.cpu_count() or 1))


def _split_env_counts(n_envs: int, num_workers: int) -> List[int]:
    base, remainder = divmod(n_envs, num_workers)
    return [base + (worker_index < remainder) for worker_index in range(num_workers)]


def _object_layout():
    offsets = {}
    offset = 0
    for name, max_count in MAX_NB_OBJECTS.items():
        offsets[name] = offset
        offset += max_count
    return offsets, set(MAX_NB_OBJECTS.keys())


def _build_single_env(render_mode: str, render_oc_overlay: bool):
    env = HackAtari(
        env_name="ALE/Seaquest-v5",
        mode="ram",
        obs_mode="ori",
        rewardfunc_path=str(REWARD_PATH),
        render_mode=render_mode,
        render_oc_overlay=render_oc_overlay,
    )
    env._env = make_env(env._env)
    return env


def _extract_logic_state(objects, offsets, relevant_objects):
    state = np.zeros((43, 4), dtype=np.float32)
    counts = {name: 0 for name in MAX_NB_OBJECTS}
    for obj in objects:
        category = obj.category
        if category not in relevant_objects:
            continue
        index = offsets[category] + counts[category]
        if category == "OxygenBar":
            state[index] = (1.0, float(obj.value), 0.0, 0.0)
        else:
            orientation = float(obj.orientation.value if obj.orientation is not None else 0)
            center_x, center_y = obj.center
            state[index] = (1.0, float(center_x), float(center_y), orientation)
        counts[category] += 1
    return state


def _worker_main(conn, env_count, render_mode, render_oc_overlay):
    offsets, relevant_objects = _object_layout()
    envs = []
    try:
        envs = [_build_single_env(render_mode, render_oc_overlay) for _ in range(env_count)]
        while True:
            command, payload = conn.recv()
            if command == "close":
                conn.send(("ok", None))
                break

            if command == "reset":
                logic_states, neural_states = [], []
                for env, seed in zip(envs, payload):
                    obs, _ = env.reset(seed=seed)
                    logic_states.append(_extract_logic_state(env.objects, offsets, relevant_objects))
                    neural_states.append(np.asarray(obs, dtype=np.float32))
                conn.send(("ok", (np.stack(logic_states), np.stack(neural_states))))
                continue

            if command == "step":
                logic_states, neural_states, rewards = [], [], []
                terminations, truncations, infos = [], [], []
                for env, action in zip(envs, payload):
                    obs, reward, terminated, truncated, info = env.step(int(action))
                    logic_states.append(_extract_logic_state(env.objects, offsets, relevant_objects))
                    neural_states.append(np.asarray(obs, dtype=np.float32))
                    rewards.append(float(reward))
                    terminations.append(bool(terminated))
                    truncations.append(bool(truncated))
                    infos.append(info)
                conn.send(
                    (
                        "ok",
                        (
                            np.stack(logic_states),
                            np.stack(neural_states),
                            np.asarray(rewards, dtype=np.float32),
                            np.asarray(terminations, dtype=np.bool_),
                            np.asarray(truncations, dtype=np.bool_),
                            infos,
                        ),
                    )
                )
                continue

            raise ValueError("Unknown command: {}".format(command))
    except Exception:
        conn.send(("error", traceback.format_exc()))
    finally:
        for env in envs:
            try:
                env.close()
            except Exception:
                pass
        conn.close()


class MultiprocessSeaquestEnv(VectorizedNudgeBaseEnv):
    """Seaquest environment with the same public contract as VectorizedNudgeEnv."""

    name = "seaquest"
    pred2action = {"noop": 0, "fire": 1, "up": 2, "right": 3, "left": 4, "down": 5}
    pred_names: Sequence

    def __init__(
        self,
        mode: str,
        n_envs: int,
        render_mode: str = "rgb_array",
        render_oc_overlay: bool = False,
        seed: Optional[int] = None,
        num_workers: int = 0,
        start_method: Optional[str] = None,
    ):
        super().__init__(mode)
        self.n_envs = n_envs
        self.n_actions = 6
        self.n_raw_actions = 18
        self.n_objects = 43
        self.n_features = 4
        self.seed = seed
        self._worker_counts = _split_env_counts(
            n_envs, _resolve_num_workers(num_workers, n_envs)
        )
        self._closed = False
        context = mp.get_context(start_method or _default_start_method())
        self._parents, self._workers = [], []
        for env_count in self._worker_counts:
            parent, child = context.Pipe()
            worker = context.Process(
                target=_worker_main,
                args=(child, env_count, render_mode, render_oc_overlay),
                daemon=True,
            )
            worker.start()
            child.close()
            self._parents.append(parent)
            self._workers.append(worker)
        atexit.register(self.close)

    def _recv(self, parent):
        status, payload = parent.recv()
        if status == "ok":
            return payload
        raise RuntimeError("Multiprocess Seaquest worker failed:\n{}".format(payload))

    def _seed_batches(self):
        if self.seed is None:
            return [[None] * count for count in self._worker_counts]
        batches, next_seed = [], self.seed
        for count in self._worker_counts:
            batches.append(list(range(next_seed, next_seed + count)))
            next_seed += count
        return batches

    def reset(self):
        for parent, seeds in zip(self._parents, self._seed_batches()):
            parent.send(("reset", seeds))
        states = [self._recv(parent) for parent in self._parents]
        return np.concatenate([state[0] for state in states]), np.concatenate([state[1] for state in states])

    def step(self, actions, is_mapped: bool = False):
        del is_mapped
        if len(actions) != self.n_envs:
            raise ValueError("Expected {} actions, received {}".format(self.n_envs, len(actions)))
        actions = np.asarray(actions)
        cursor = 0
        for parent, count in zip(self._parents, self._worker_counts):
            parent.send(("step", actions[cursor : cursor + count]))
            cursor += count
        results = [self._recv(parent) for parent in self._parents]
        return (
            (
                np.concatenate([result[0] for result in results]),
                np.concatenate([result[1] for result in results]),
            ),
            np.concatenate([result[2] for result in results]),
            np.concatenate([result[3] for result in results]),
            np.concatenate([result[4] for result in results]),
            [info for result in results for info in result[5]],
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        for parent in self._parents:
            try:
                parent.send(("close", None))
            except Exception:
                pass
        for parent in self._parents:
            try:
                parent.recv()
            except Exception:
                pass
            parent.close()
        for worker in self._workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=1)
        self._parents.clear()
        self._workers.clear()
