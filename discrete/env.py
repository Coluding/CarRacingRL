import torch
import numpy as np
import gym
import cv2 as cv
from typing import Union, Sequence
import collections


class RepeatActionAndMaxFrame(gym.Wrapper):
    """
    This wrapper repeats the action for a number of steps and returns the maximum frame from the last two frames.
    This is used to handle the flickering issue in atari games.
    """

    def __init__(self, env: gym.Env, repeat: int = 4, frame_stack: int = 2, no_ops: int = 50):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.shape
        self.frame_buffer = np.zeros((frame_stack, *self.shape))
        self.frame_stack = frame_stack
        self.skip_start_steps = no_ops

        if self.repeat < 1:
            raise ValueError("Repeat value must be greater than 0")

    def step(self, action: Union[int, float]):
        t_reward = 0
        done = False

        for i in range(self.repeat):
            obs, reward, done, trunc, info = self.env.step(action)

            if not isinstance(obs, np.ndarray):
                obs = obs[0]

            idx = i % self.frame_stack
            self.frame_buffer[idx] = obs
            t_reward += reward

            if done:
                break

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])

        return max_frame, t_reward, done, trunc, info

    def reset(self):
        """
        In some games, the agent is required to perform a number of no-op actions before the game starts. This method
        performs the no-op actions and returns the initial observation. For example, in Car-racing the game screen is
        slowly zoomed in and the agent is required to perform no-op actions until the screen is fully zoomed in.
        :return:
        """
        obs = self.env.reset()

        for _ in range(self.skip_start_steps):
            obs, _, _, _, _ = self.env.step(0)

        if not isinstance(obs, np.ndarray):
            obs = obs[0]

        for i in range(self.frame_stack):
            self.frame_buffer[i] = obs

        return self.frame_buffer[0], {}


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, new_shape: Sequence[int] = (84, 84, 1)):
        super(PreprocessFrame, self).__init__(env)
        self.shape = new_shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)

    def observation(self, obs: np.ndarray):
        obs = obs.astype(np.uint8)
        new_frame = cv.cvtColor(obs, cv.COLOR_RGB2GRAY)
        new_frame = cv.resize(new_frame, self.shape[1:], interpolation=cv.INTER_AREA)
        new_frame = np.array(new_frame, dtype=np.uint8).reshape(self.shape)

        new_frame = new_frame / 255.0
        return new_frame


class StackFrames(gym.ObservationWrapper):
    """
    StackFrames is a wrapper that stacks frames together to form a new observation space. This is the wrapper that
    actually stacks the frames together to form a new observation space. The observation space is a Box space with
    the low and high values being the low and high values of the environment's observation space repeated by the
    number of frames to stack.
    """

    def __init__(self, env: gym.Env, repeat: int = 4):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(env.observation_space.low.repeat(repeat, axis=0),
                                                env.observation_space.high.repeat(repeat, axis=0), dtype=np.float32)
        self.stack = collections.deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()
        obs, _ = self.env.reset()

        for _ in range(self.stack.maxlen):
            self.stack.append(obs)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, obs: np.ndarray):
        """
        We just push the observation into the stack and return the stack reshaped to the observation space low shape.
        :param obs:
        :return:
        """
        self.stack.append(obs)
        return np.array(self.stack).reshape(self.observation_space.low.shape)


def make_env(env_name, shape=(84, 84, 1), repeat=4, clip_rewards=False, no_ops=45, fire_first=False, **kwargs) -> gym.Env:
    env = gym.make(env_name, render_mode='rgb_array', **kwargs)
    env = RepeatActionAndMaxFrame(env, repeat, no_ops=no_ops)
    env = PreprocessFrame(env, shape)
    env = StackFrames(env, repeat)

    return env
