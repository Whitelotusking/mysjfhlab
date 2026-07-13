import gymnasium as gym

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


def make_env(env):
    env = gym.wrappers.RecordEpisodeStatistics(env)
    autoreset_wrapper = getattr(gym.wrappers, "Autoreset", None)
    if autoreset_wrapper is None:
        autoreset_wrapper = gym.wrappers.AutoResetWrapper
    grayscale_wrapper = getattr(gym.wrappers, "GrayscaleObservation", None)
    if grayscale_wrapper is None:
        grayscale_wrapper = gym.wrappers.GrayScaleObservation
    frame_stack_wrapper = getattr(gym.wrappers, "FrameStackObservation", None)
    if frame_stack_wrapper is None:
        frame_stack_wrapper = gym.wrappers.FrameStack
    env = autoreset_wrapper(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = grayscale_wrapper(env)
    env = frame_stack_wrapper(env, 4)
    return env


kangaroo_modifs = [
    "disable_coconut",
    "randomize_kangaroo_position",
    "change_level_0",
]

seaquest_modifs = []
