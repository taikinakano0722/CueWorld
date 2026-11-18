import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from envs.communication import CommunicationEnv


def test_gymnasium_env_checker_passes():
    env = CommunicationEnv(max_steps=8, seed=0)
    check_env(env, skip_render_check=True)


def test_rejects_invalid_constructor_args():
    with pytest.raises(ValueError):
        CommunicationEnv(max_steps=0)
    with pytest.raises(ValueError):
        CommunicationEnv(max_steps=8, n_meaningful=0)
    with pytest.raises(ValueError):
        CommunicationEnv(max_steps=8, n_meaningful=5)
    with pytest.raises(ValueError):
        CommunicationEnv(max_steps=8, obs_noise=-0.01)
    with pytest.raises(ValueError):
        CommunicationEnv(max_steps=8, dependency="xxx")


def test_reset_obs_and_info():
    env = CommunicationEnv(max_steps=4, seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (4,)
    assert info["parent_symbol"] in range(4)
    assert info["dependency"] == "short"
    assert info["phase"] == "outside"
    assert info["yoke"] is False
    assert info["n_meaningful"] == 1
    expected = np.zeros(4, dtype=np.float32)
    expected[info["parent_symbol"]] = 1.0
    assert np.allclose(obs, expected)


def test_short_meaningful_reply():
    env = CommunicationEnv(max_steps=20, seed=0, n_meaningful=1)
    env.reset(seed=0)

    # action 0 is meaningful → parent 2
    for _ in range(10):
        obs, reward, _, _, info = env.step(0)
        assert reward == 0.0
        assert info["parent_symbol"] == 2
        expected = np.zeros(4, dtype=np.float32)
        expected[2] = 1.0
        assert np.allclose(obs, expected)


def test_short_all_meaningful():
    env = CommunicationEnv(max_steps=20, seed=0, n_meaningful=4)
    env.reset(seed=0)
    expected_replies = {0: 2, 1: 3, 2: 0, 3: 1}
    for action, reply in expected_replies.items():
        obs, _, _, _, info = env.step(action)
        assert info["parent_symbol"] == reply
        expected = np.zeros(4, dtype=np.float32)
        expected[reply] = 1.0
        assert np.allclose(obs, expected)


def test_short_unmeaningful_is_uniform():
    env = CommunicationEnv(max_steps=200, seed=0, n_meaningful=1)
    env.reset(seed=0)
    symbols = []
    for _ in range(200):
        _, _, _, _, info = env.step(1)  # not meaningful when n_meaningful=1
        symbols.append(info["parent_symbol"])
    assert set(symbols) == {0, 1, 2, 3}


def test_long_chain_a():
    env = CommunicationEnv(max_steps=10, seed=0, dependency="long")
    env.reset(seed=0)

    obs, _, _, _, info = env.step(0)
    assert info["parent_symbol"] == 1
    assert info["phase"] == "mid_a"

    obs, _, _, _, info = env.step(2)
    assert info["parent_symbol"] == 3
    assert info["phase"] == "outside"


def test_long_chain_b():
    env = CommunicationEnv(max_steps=10, seed=0, dependency="long")
    env.reset(seed=0)

    _, _, _, _, info = env.step(1)
    assert info["parent_symbol"] == 0
    assert info["phase"] == "mid_b"

    _, _, _, _, info = env.step(3)
    assert info["parent_symbol"] == 2
    assert info["phase"] == "outside"


def test_long_mid_wrong_continuation():
    env = CommunicationEnv(max_steps=20, seed=0, dependency="long")
    env.reset(seed=0)

    # Start chain A
    _, _, _, _, info = env.step(0)
    assert info["phase"] == "mid_a"

    # Wrong continuation: action 3 during mid_a → uniform, then back to outside
    symbols = []
    for _ in range(40):
        env2 = CommunicationEnv(max_steps=4, seed=_, dependency="long")
        env2.reset(seed=_)
        env2.step(0)  # enter mid_a
        _, _, _, _, info = env2.step(3)  # wrong
        symbols.append(info["parent_symbol"])
        assert info["phase"] == "outside"
    assert set(symbols) == {0, 1, 2, 3}

    # After returning to outside, action 1 starts chain B
    env.reset(seed=0)
    env.step(0)
    env.step(3)  # wrong mid, back to outside
    _, _, _, _, info = env.step(1)
    assert info["parent_symbol"] == 0
    assert info["phase"] == "mid_b"


def test_long_start_during_mid():
    """途中の 0 や 1 は新しい鎖の開始にならない。"""
    symbols = []
    for seed in range(50):
        env = CommunicationEnv(max_steps=4, seed=seed, dependency="long")
        env.reset(seed=seed)
        env.step(0)  # enter mid_a
        _, _, _, _, info = env.step(0)  # mid での 0 → 一様、開始にならない
        symbols.append(info["parent_symbol"])
        assert info["phase"] == "outside"
    assert set(symbols) == {0, 1, 2, 3}


def test_long_outside_2_and_3_uniform():
    symbols = []
    for seed in range(50):
        env = CommunicationEnv(max_steps=4, seed=seed, dependency="long")
        env.reset(seed=seed)
        _, _, _, _, info = env.step(2)
        symbols.append(info["parent_symbol"])
        assert info["phase"] == "outside"
    assert set(symbols) == {0, 1, 2, 3}


def test_yoke_replays_symbols():
    env = CommunicationEnv(max_steps=6, seed=0, n_meaningful=1)
    env.reset(seed=0)

    recorded = []
    infant_actions = [0, 0, 1, 1, 2, 3]
    for a in infant_actions:
        _, _, _, truncated, info = env.step(a)
        recorded.append(info["parent_symbol"])
        if truncated:
            break
    tape = np.asarray(recorded, dtype=np.int64)
    assert tape.shape == (6,)
    # First two actions are meaningful → parent 2
    assert tape[0] == 2 and tape[1] == 2

    # Yoke with different infant actions: parent tape is independent
    obs0, info0 = env.reset(options={"yoke_parent_symbols": tape})
    assert info0["yoke"] is True
    assert info0["parent_symbol"] in range(4)

    yoke_infant = [3, 3, 3, 3, 3, 3]
    for i, a in enumerate(yoke_infant):
        obs, reward, terminated, truncated, info = env.step(a)
        assert reward == 0.0
        assert terminated is False
        assert info["yoke"] is True
        assert info["parent_symbol"] == int(tape[i])
        expected = np.zeros(4, dtype=np.float32)
        expected[int(tape[i])] = 1.0
        assert np.allclose(obs, expected)
        if i + 1 == env.max_steps:
            assert truncated is True
        else:
            assert truncated is False


def test_yoke_shape_validation():
    env = CommunicationEnv(max_steps=3, seed=0)
    with pytest.raises(ValueError):
        env.reset(options={"yoke_parent_symbols": np.zeros(2, dtype=np.int64)})
    with pytest.raises(ValueError):
        env.reset(options={"yoke_parent_symbols": np.zeros((3, 1), dtype=np.int64)})
    with pytest.raises(ValueError):
        env.reset(options={"yoke_parent_symbols": np.array([0, 1, 4], dtype=np.int64)})


def test_yoke_noise_is_fresh():
    tape = np.array([0, 1, 2, 3], dtype=np.int64)
    env = CommunicationEnv(max_steps=4, obs_noise=0.5, seed=0)
    env.reset(seed=0, options={"yoke_parent_symbols": tape})
    obs1 = []
    for a in range(4):
        obs, _, _, _, _ = env.step(a)
        obs1.append(obs.copy())

    env2 = CommunicationEnv(max_steps=4, obs_noise=0.5, seed=1)
    env2.reset(seed=1, options={"yoke_parent_symbols": tape})
    obs2 = []
    for a in range(4):
        obs, _, _, _, _ = env2.step(a)
        obs2.append(obs.copy())

    # Same clean symbols, but noise realizations differ
    assert not all(np.allclose(a, b) for a, b in zip(obs1, obs2))


def test_reward_always_zero():
    env = CommunicationEnv(max_steps=20, seed=0, n_meaningful=4)
    env.reset(seed=0)
    for action in range(4):
        _, reward, _, _, _ = env.step(action)
        assert reward == 0.0


def test_truncation_at_max_steps():
    env = CommunicationEnv(max_steps=2, seed=0)
    env.reset(seed=0)
    env.step(0)
    _, _, terminated, truncated, _ = env.step(0)
    assert terminated is False
    assert truncated is True
    with pytest.raises(RuntimeError):
        env.step(0)


def test_obs_noise():
    env = CommunicationEnv(max_steps=10, obs_noise=0.25, seed=0, n_meaningful=1)
    env.reset(seed=0)
    obs, _, _, _, info = env.step(0)
    assert info["parent_symbol"] == 2
    assert not np.allclose(obs, [0.0, 0.0, 1.0, 0.0])


def test_reset_obs_is_uniform():
    symbols = []
    for seed in range(200):
        env = CommunicationEnv(max_steps=4, seed=seed)
        _, info = env.reset(seed=seed)
        symbols.append(info["parent_symbol"])
    assert set(symbols) == {0, 1, 2, 3}


def test_info_parent_symbol():
    env = CommunicationEnv(max_steps=20, seed=0, n_meaningful=4, obs_noise=0.0)
    env.reset(seed=0)
    for action in range(4):
        obs, _, _, _, info = env.step(action)
        assert info["parent_symbol"] in range(4)
        assert int(np.argmax(obs)) == info["parent_symbol"]
        assert abs(float(obs[info["parent_symbol"]]) - 1.0) < 1e-6
