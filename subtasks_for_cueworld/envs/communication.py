from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CommunicationEnv(gym.Env):
    """
    幼児の信号に対して親が返事を返すコミュニケーション環境。

    - 行動は信号 4 種。何もしないは置かない。エネルギーは置かない。
    - step(a) が返す観測は、a への親の信号（長さ 4 の one-hot）。
    - reward は常に 0。学習するエージェントは置かない。
    - dependency="short": 行動から親記号への長さ 1 の対応。
    - dependency="long": 鎖の外 / 途中の位相を持つ長期依存。
    """

    metadata = {"render_modes": []}

    # 短期の対応表。番号順に n_meaningful 個だけ有効。
    REPLY_TABLE = {
        0: 2,
        1: 3,
        2: 0,
        3: 1,
    }

    N_SYMBOLS = 4
    N_ACTIONS = 4
    OBS_DIM = 4
    PHASE_OUTSIDE = "outside"
    PHASE_MID_A = "mid_a"
    PHASE_MID_B = "mid_b"

    def __init__(
        self,
        *,
        max_steps: int,
        n_meaningful: int = 1,
        dependency: str = "short",
        obs_noise: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        if not 1 <= n_meaningful <= self.N_ACTIONS:
            raise ValueError(
                f"n_meaningful must be in [1, {self.N_ACTIONS}], got {n_meaningful}"
            )
        if dependency not in ("short", "long"):
            raise ValueError(
                f'dependency must be "short" or "long", got {dependency!r}'
            )
        if obs_noise < 0.0:
            raise ValueError(f"obs_noise must be >= 0, got {obs_noise}")

        self.max_steps = int(max_steps)
        self.n_meaningful = int(n_meaningful)
        self.dependency = dependency
        self.obs_noise = float(obs_noise)

        # 短期で有効なトリガ行動（表の先頭から n_meaningful 個）
        self._meaningful_actions = set(range(self.n_meaningful))

        self.action_space = spaces.Discrete(self.N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.OBS_DIM,),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._episode_started = False
        self._phase = self.PHASE_OUTSIDE
        self._yoke_symbols: Optional[np.ndarray] = None
        self._yoke_index = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._episode_started = True
        self._phase = self.PHASE_OUTSIDE
        self._yoke_index = 0
        self._yoke_symbols = None

        if options is not None and "yoke_parent_symbols" in options:
            tape = np.asarray(options["yoke_parent_symbols"])
            expected = (self.max_steps,)
            if tape.shape != expected:
                raise ValueError(
                    f"yoke_parent_symbols shape must be {expected}, got {tape.shape}"
                )
            if tape.dtype.kind not in ("i", "u"):
                tape = tape.astype(np.int64)
            if np.any(tape < 0) or np.any(tape >= self.N_SYMBOLS):
                raise ValueError(
                    f"yoke_parent_symbols values must be in [0, {self.N_SYMBOLS - 1}]"
                )
            self._yoke_symbols = tape.astype(np.int64, copy=True)

        # reset の観測は直前の行動が無いので 4 記号から一様。yoke の列には含めない。
        parent_symbol = int(self._rng.integers(0, self.N_SYMBOLS))
        obs = self._build_obs(parent_symbol)
        return obs, self._base_info(parent_symbol=parent_symbol)

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self._episode_started:
            raise RuntimeError("Call reset before step")
        if self._step_count >= self.max_steps:
            raise RuntimeError("Episode already truncated; call reset")

        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"action out of range: {action}")

        if self._yoke_symbols is not None:
            parent_symbol = int(self._yoke_symbols[self._yoke_index])
            self._yoke_index += 1
            # yoke 中は位相を進めない
        elif self.dependency == "short":
            parent_symbol = self._sample_short(action)
        else:
            parent_symbol = self._sample_long(action)

        obs = self._build_obs(parent_symbol)

        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.max_steps
        if truncated:
            self._episode_started = False

        info = self._base_info(parent_symbol=parent_symbol)
        return obs, 0.0, terminated, truncated, info

    def _sample_short(self, action: int) -> int:
        if action in self._meaningful_actions:
            return self.REPLY_TABLE[action]
        return int(self._rng.integers(0, self.N_SYMBOLS))

    def _sample_long(self, action: int) -> int:
        if self._phase == self.PHASE_OUTSIDE:
            if action == 0:
                self._phase = self.PHASE_MID_A
                return 1
            if action == 1:
                self._phase = self.PHASE_MID_B
                return 0
            # 2, 3: 一様、位相は外のまま
            return int(self._rng.integers(0, self.N_SYMBOLS))

        if self._phase == self.PHASE_MID_A:
            # 途中は 1 手だけ。合否にかかわらず外に戻る。
            self._phase = self.PHASE_OUTSIDE
            if action == 2:
                return 3
            return int(self._rng.integers(0, self.N_SYMBOLS))

        # PHASE_MID_B
        self._phase = self.PHASE_OUTSIDE
        if action == 3:
            return 2
        return int(self._rng.integers(0, self.N_SYMBOLS))

    def _build_obs(self, parent_symbol: int) -> np.ndarray:
        obs = np.zeros(self.N_SYMBOLS, dtype=np.float32)
        obs[parent_symbol] = 1.0
        if self.obs_noise > 0.0:
            obs = obs + self._rng.normal(
                0.0, self.obs_noise, size=self.N_SYMBOLS
            ).astype(np.float32)
        return obs.astype(np.float32, copy=False)

    def _base_info(self, *, parent_symbol: int) -> dict[str, Any]:
        return {
            "parent_symbol": parent_symbol,
            "dependency": self.dependency,
            "phase": self._phase,
            "yoke": self._yoke_symbols is not None,
            "n_meaningful": self.n_meaningful,
        }
