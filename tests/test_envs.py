import pytest
from tensordict import TensorDict
import torch
from torchrl.envs.utils import check_env_specs, step_mdp
from lerobot.common.datasets.factory import make_dataset
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from lerobot.common.envs.aloha.env import AlohaEnv
from lerobot.common.envs.factory import make_env
from lerobot.common.envs.pusht.env import PushtEnv
from lerobot.common.envs.simxarm.env import SimxarmEnv
from lerobot.common.utils import init_hydra_config

from .utils import DEVICE, DEFAULT_CONFIG_PATH


def print_spec_rollout(env):
    print("observation_spec:", env.observation_spec)
    print("action_spec:", env.action_spec)
    print("reward_spec:", env.reward_spec)
    print("done_spec:", env.done_spec)

    td = env.reset()
    print("reset tensordict", td)

    td = env.rand_step(td)
    print("random step tensordict", td)

    def simple_rollout(steps=100):
        # preallocate:
        data = TensorDict({}, [steps])
        # reset
        _data = env.reset()
        for i in range(steps):
            _data["action"] = env.action_spec.rand()
            _data = env.step(_data)
            data[i] = _data
            _data = step_mdp(_data, keep_other=True)
        return data

    print("data from rollout:", simple_rollout(100))


@pytest.mark.parametrize(
    "task,from_pixels,pixels_only",
    [
        ("sim_insertion", True, False),
        ("sim_insertion", True, True),
        ("sim_transfer_cube", True, False),
        ("sim_transfer_cube", True, True),
    ],
)
def test_aloha(task, from_pixels, pixels_only):
    env = AlohaEnv(
        task,
        from_pixels=from_pixels,
        pixels_only=pixels_only,
        image_size=[3, 480, 640] if from_pixels else None,
    )
    # print_spec_rollout(env)
    check_env_specs(env)


@pytest.mark.parametrize(
    "task, obs_type",
    [
        ("XarmLift-v0", "state"),
        ("XarmLift-v0", "pixels"),
        ("XarmLift-v0", "pixels_agent_pos"),
        # TODO(aliberts): Add simxarm other tasks
    ],
)
def test_xarm(env_task, obs_type):
    import gym_xarm
    env = gym.make(f"gym_xarm/{env_task}", obs_type=obs_type)
    # env = SimxarmEnv(
    #     task,
    #     from_pixels=from_pixels,
    #     pixels_only=pixels_only,
    #     image_size=84 if from_pixels else None,
    # )
    # print_spec_rollout(env)
    # check_env_specs(env)
    check_env(env)


@pytest.mark.parametrize(
    "from_pixels,pixels_only",
    [
        (True, False),
    ],
)
def test_pusht(from_pixels, pixels_only):
    env = PushtEnv(
        from_pixels=from_pixels,
        pixels_only=pixels_only,
        image_size=96 if from_pixels else None,
    )
    # print_spec_rollout(env)
    check_env_specs(env)


@pytest.mark.parametrize(
    "env_name",
    [
        "simxarm",
        "pusht",
        "aloha",
    ],
)
def test_factory(env_name):
    cfg = init_hydra_config(
        DEFAULT_CONFIG_PATH,
        overrides=[f"env={env_name}", f"device={DEVICE}"],
    )

    dataset = make_dataset(cfg)

    env = make_env(cfg)
    for key in dataset.image_keys:
        assert env.reset().get(key).dtype == torch.uint8
    check_env_specs(env)

    env = make_env(cfg, transform=dataset.transform)
    for key in dataset.image_keys:
        img = env.reset().get(key)
        assert img.dtype == torch.float32
        # TODO(rcadene): we assume for now that image normalization takes place in the model
        assert img.max() <= 1.0
        assert img.min() >= 0.0
    check_env_specs(env)
