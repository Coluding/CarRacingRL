import gym
import os
import numpy as np
import torch
from enum import Enum
import logging
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from torch.utils.tensorboard import SummaryWriter
from typing import Union
from env import make_env
from agent import AbstractAgent, DDQNAgent
from utils import prep_logger


class TrainingAlgorithms(Enum):
    DDQN = 1
    PPO = 2


class RLTrainer:
    def __init__(self, agent: AbstractAgent, env: gym.Env, summary_writer: SummaryWriter, n_games: int,
                 start_counter: int = 50, update_step: int = 2, render_interval: int = 100,
                 load_checkpoint: bool = False):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.agent = agent
        if load_checkpoint:
            self.load_models()
        self.env = env
        self.n_games = n_games
        self.update_step = update_step
        self.render_interval = render_interval
        self.summary_writer = summary_writer
        self.start_counter = start_counter
        self.best_score = -np.inf

        self._create_directories()
        self.logger.info("RLTrainer initialized")

    def _create_directories(self):
        os.makedirs('models', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        os.makedirs('video', exist_ok=True)
        self.logger.info("Directories for models, plots, and videos created")

    def train(self, training_algorithm: TrainingAlgorithms = TrainingAlgorithms.DDQN):
        n_steps = 0
        scores, eps_history, steps_array = [], [], []
        best_score = -np.inf

        for i in range(self.n_games):
            self.logger.info(f"Starting game {i + 1}/{self.n_games}")
            observation = self.env.reset()
            if not isinstance(observation, np.ndarray):
                observation = observation[0]

            score = 0
            done = False
            while not done:
                if training_algorithm == training_algorithm.DDQN:
                    action = self.agent.choose_action(observation)
                    observation_, reward, done, trunc, info = self.env.step(action)
                    self.agent.add_experience(observation, action, done, reward, observation_)

                    if n_steps % self.update_step == 0:
                        self.agent.learn()

                    observation = observation_
                    score += reward
                    n_steps += 1

                elif training_algorithm == training_algorithm.PPO:
                    action, value, proba = self.agent.choose_action(observation)
                    observation_, reward, done, trunc, info = self.env.step(action)
                    self.agent.add_experience(observation, action, reward, done, proba, value)

                    if n_steps % self.update_step == 0:
                        self.agent.learn()

                    observation = observation_

                if i % self.render_interval == 0 and not done and i != 0:
                    self.logger.info(f"Rendering video at game {i + 1}")
                    self.eval_video(i)

            self.summary_writer.add_scalar('Score', score, i)
            scores.append(score)
            steps_array.append(n_steps)
            self.summary_writer.add_scalar('Steps', n_steps, i)
            avg_score = np.mean(scores[-100:])
            self.summary_writer.add_scalar('Average Score', avg_score, i)
            self.logger.info(f"Game {i + 1}: score {score} ----- average score {avg_score} ----- {n_steps} steps")

            if avg_score > best_score:
                best_score = avg_score
                self.save_models()

    def adjust_epsilon(self, epsilon: float):
        self.agent.epsilon = epsilon

    def save_models(self):
        self.agent.save_models()
        self.logger.info("Models saved")

    def load_models(self):
        self.agent.load_models()
        self.logger.info("Models loaded")

    def eval_video(self, global_step: Union[int, None] = None, steps: int = 1000):
        self.logger.info("Recording evaluation video")
        video = VideoRecorder(self.env, f"video/evaluation_{global_step}.mp4")
        observation = self.env.reset()
        if not isinstance(observation, np.ndarray):
            observation = observation[0]

        step_counter = 0
        done = False
        score = 0
        while not done and step_counter < steps:
            self.env.render()
            action = self.agent.choose_action(observation, deterministic=True)
            observation_, reward, done, trunc, info = self.env.step(action)
            video.capture_frame()
            observation = observation_
            score += reward
            step_counter += 1

        if global_step is not None:
            self.summary_writer.add_scalar('Evaluation Score', score, global_step)

        video.close()
        self.logger.info(f"Recorded evaluation video at step {global_step}")


if __name__ == '__main__':
    # Configure logging
    logger = prep_logger()
    env_shape = (1, 84, 84)
    model_shape = (4, 84, 84)
    repeats = 4
    env = make_env('ALE/Pacman-v5', env_shape, repeats, continuous=False, no_ops=1)
    summary_writer = SummaryWriter()
    agent = DDQNAgent(alpha=0.0001, gamma=0.99, cnn_channel=[128, 256, 512], input_dims=model_shape,
                      fc_1_dims=512, fc_2_dims=512, n_actions=4, eps=1.0, eps_min=0.1, eps_dec=1e-4,
                      mem_size=50000, batch_size=200, prioritized_replay=True, replace_target=1000,
                      chkpt_dir="models/ddqn_atlantis", summary_writer=summary_writer)

    trainer = RLTrainer(agent, env, summary_writer, n_games=300, update_step=2, render_interval=25)
    #trainer.load_models()
    #trainer.eval_video(11)
    #trainer.adjust_epsilon(0.9)
    try:
        trainer.train()

    except KeyboardInterrupt:
        trainer.save_models()
        trainer.eval_video(11)
        env.close()
        logger.info("Training interrupted, models saved and video recorded")
        raise