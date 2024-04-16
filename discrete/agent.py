import torch
import numpy as np
from typing import Sequence, Union, Tuple
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
from tqdm import tqdm
from car_racing.discrete.replay_buffer import ReplayBuffer, OnPolicyMemory
from car_racing.discrete.model import DQN, ActorNetwork, CriticNetwork
from car_racing.discrete.utils import convert_tuple_output
import os


class AbstractAgent(ABC):
    @abstractmethod
    def choose_action(self, observation: np.ndarray, deterministic: bool = True) -> Union[int, Tuple[int, ...]]:
        pass

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def replace_target_network(self):
        pass

    @abstractmethod
    def add_experience(self, state: np.ndarray, action: int,
                       done: bool, reward: float, state_: np.ndarray,
                       priority: bool = True):
        pass

    @abstractmethod
    def save_models(self):
        pass

    @abstractmethod
    def load_models(self):
        pass


class DDQNAgent(AbstractAgent):
    def __init__(self, alpha: float, gamma: float, cnn_channel: Sequence[int], input_dims: Sequence[int],
                 fc_1_dims: int, fc_2_dims: int, n_actions: int, eps: float, eps_min: float, eps_dec: float,
                 mem_size: int, batch_size: int, prioritized_replay: bool = False,
                 replace_target: int = 1000, chkpt_dir="tmp/ddqn",
                 summary_writer: SummaryWriter = None):

        self.gamma = gamma
        self.epsilon = eps
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.batch_size = batch_size
        self.replace_target = replace_target
        self.chkpt_dir = chkpt_dir
        self.summary_writer = summary_writer

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, prioritized_replay=prioritized_replay)
        self.q_eval = DQN(alpha, n_actions, input_dims, fc_1_dims, fc_2_dims,
                          cnn_channel, name='q_eval', chkpt_dir=chkpt_dir)

        self.q_target1 = DQN(alpha, n_actions, input_dims, fc_1_dims, fc_2_dims,
                             cnn_channel, name='q_target', chkpt_dir=chkpt_dir)

        self.q_target2 = DQN(alpha, n_actions, input_dims, fc_1_dims, fc_2_dims,
                             cnn_channel, name='q_target', chkpt_dir=chkpt_dir)

    def choose_action(self, observation: np.ndarray, deterministic: bool = False) -> int:
        if np.random.random() < self.epsilon and not deterministic:
            action = np.random.choice(self.action_space)
        else:
            state = torch.tensor([observation], dtype=torch.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = torch.argmax(actions).item()
        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target == 0:
            self.q_target1.load_state_dict(self.q_eval.state_dict())
            self.q_target2.load_state_dict(self.q_eval.state_dict())

    def add_experience(self, state: np.ndarray, action: int,
                       done: bool, reward: float, state_: np.ndarray,
                       priority: bool = True):
        if priority:
            error = self.compute_td_error(state, action, reward, state_, done, grad=False)
            self.memory.add_experience(state, action, done, reward, state_, error.item())
        else:
            self.memory.add_experience(state, action, done, reward, state_)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        states, actions, rewards, states_, dones, indices = self.memory.sample_buffer(self.batch_size)

        loss = self.compute_td_error(states, actions, rewards, states_, dones, grad=True)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.summary_writer.add_scalar('Loss', loss, self.learn_step_counter)
        self.summary_writer.add_scalar('Epsilon', self.epsilon, self.learn_step_counter)

        self.decrement_epsilon()

    def compute_td_error(self, states: Union[np.ndarray, torch.Tensor], action: Union[np.ndarray, torch.Tensor, int],
                         rewards: Union[np.ndarray, torch.Tensor, float], states_: Union[np.ndarray, torch.Tensor],
                         dones: Union[np.ndarray, torch.Tensor, bool], grad: bool = True) -> torch.Tensor:

        if not isinstance(states, torch.Tensor):
            rewards = torch.tensor(rewards).to(self.q_eval.device)
            states = torch.tensor(states, dtype=torch.float32).to(self.q_eval.device)
            if len(states.shape) != 4:
                states = states.unsqueeze(0)
            states_ = torch.tensor(states_, dtype=torch.float32).to(self.q_eval.device)
            if len(states_.shape) != 4:
                states_ = states_.unsqueeze(0)

        if grad:
            q_pred = self.q_eval.forward(states)[np.arange(states.shape[0]), action]
            q_next1 = self.q_target1.forward(states_)
            q_next2 = self.q_target2.forward(states_)
            q_value_next = torch.min(torch.max(q_next1, dim=1)[0], torch.max(q_next2, dim=1)[0])

            q_target = rewards + self.gamma * q_value_next
            q_target[dones] = 0.0
            error = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)

        else:
            with torch.no_grad():
                q_pred = self.q_eval.forward(states)[np.arange(states.shape[0]), action]
                q_next1 = self.q_target1.forward(states_)
                q_next2 = self.q_target2.forward(states_)

                q_value_next = torch.min(torch.max(q_next1, dim=1)[0], torch.max(q_next2, dim=1)[0])

                q_target = rewards + self.gamma * q_value_next
                q_target[dones] = 0.0
                error = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)

        return error

    def save_models(self):
        if os.path.exists(self.chkpt_dir) is False:
            os.makedirs(self.chkpt_dir)

        self.q_eval.save_checkpoint()
        self.q_target1.save_checkpoint()
        self.q_target2.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_target1.load_checkpoint()
        self.q_target2.load_checkpoint()


class PPOAgent(AbstractAgent):
    def __init__(self, alpha: float, gamma: float, gae_lambda: float, clip_epsilon: float,
                 cnn_channel: Sequence[int], input_dims: Sequence[int],
                 fc_1_dims: int, fc_2_dims: int, n_actions: int, eps: float, eps_min: float, eps_dec: float,
                 batch_size: int, replace_target: int = 1000, chkpt_dir="tmp/ppo",
                 summary_writer: SummaryWriter = None):
        self.gamma = gamma
        self.epsilon = eps
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.batch_size = batch_size
        self.replace_target = replace_target
        self.chkpt_dir = chkpt_dir
        self.summary_writer = summary_writer

        self.memory = OnPolicyMemory(batch_size)
        self.actor = ActorNetwork(alpha, input_dims, fc_1_dims, fc_2_dims, n_actions,
                                  cnn_channel, chkpt_dir=chkpt_dir)
        self.critic = CriticNetwork

    def add_experience(self, state: np.ndarray, action: int,
                       done: bool, reward: float, state_: np.ndarray,
                       priority: bool = True):
        self.memory.store_memory(state, action, reward, state_, done)

    def learn(self, entropy_regularization: float = .0):
        states, actions, probs, values, rewards, dones, batches = self.memory.generate_batches()
        iterator = tqdm(batches, total=len(batches))

        advantage: np.ndarray = self._compute_gae_advantage(rewards, values, dones)

        for batch in iterator:
            states_tensor = torch.tensor(states[batch], dtype=torch.float).to(self.actor.device)
            old_log_probs_tensor = torch.tensor(probs[batch], dtype=torch.float).to(self.actor.device)

            value_estimate: torch.Tensor = self.critic.forward(states_tensor)
            returns = advantage[batch] + values[batch]
            dist = self.actor.forward(states_tensor)
            log_prob = dist.log_prob(torch.tensor(actions).to(self.actor.device))
            ratio = torch.exp(log_prob - old_log_probs_tensor)

            advantage_tensor = torch.tensor(advantage, dtype=torch.float32, device=self.actor.device)
            clipped_probs = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            objective = torch.min(ratio * advantage_tensor,
                                  clipped_probs * advantage_tensor)

            objective += entropy_regularization * self._compute_entropy(log_prob)

            actor_loss = -torch.mean(objective)

            critic_loss = torch.mean((torch.tensor(returns, dtype=torch.float32, device=self.actor.device)
                                      - value_estimate) ** 2)

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

    def _compute_entropy(self, log_probs: torch.Tensor) -> torch.Tensor:
        return -torch.sum(log_probs.exp() * log_probs)

    def _compute_gae_advantage(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray):

        advantages = np.zeros(len(rewards), dtype=np.float32)

        last_advantage = 0
        for t in reversed(range(len(rewards))):
            terminal = 1 - int(dones[t])
            next_value = values[t + 1] if not t == len(rewards) else 0
            delta = terminal * (rewards[t] + self.gamma * next_value) - values[t]
            advantages[t] = self.gae_lambda * self.gamma * last_advantage + delta
            last_advantage = advantages[t]

        return advantages

    def choose_action(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, ...]:
        state_tensor = torch.tensor(observation, dtype=torch.float32, device=self.actor.device)
        dist: torch.distributions.categorical.Categorical = self.actor(state_tensor)
        value: torch.Tensor = self.critic.forward(state_tensor)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return value.item(), action.item(), log_prob.item()


    def save_models(self):
        if os.path.exists(self.chkpt_dir) is False:
            os.makedirs(self.chkpt_dir)

        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()


    def replace_target_network(self):
        pass