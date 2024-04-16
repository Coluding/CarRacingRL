import torch
import numpy as np
from typing import Union, Sequence

class ReplayBuffer:
    def __init__(self, max_size: int, input_shape: Union[Sequence, int],
                 n_actions: int, prioritized_replay: bool = False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.prioritized_replay = prioritized_replay
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)
        self.priorities = np.zeros(self.mem_size, dtype=np.float32) if prioritized_replay else None

    def add_experience(self, state: np.ndarray, action: int, done: bool,
                       reward: float, state_: np.ndarray, priority: float = 1.0):
        idx = self.mem_cntr % self.mem_size
        self.state_memory[idx] = state
        self.new_state_memory[idx] = state_
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = done

        if self.prioritized_replay:
            self.priorities[idx] = priority

        self.mem_cntr += 1

    def _recompute_priorities(self, beta):
        max_mem = min(self.mem_cntr, self.mem_size)
        priorities = self.priorities[:max_mem]
        probabilities = priorities ** beta
        probabilities /= probabilities.sum()
        return probabilities

    def sample_buffer(self, batch_size: int, beta: float = 0.4):
        if self.prioritized_replay:
            probas = self._recompute_priorities(beta)

            indices = np.random.choice(np.arange(min(self.mem_cntr, self.mem_size)),
                                       batch_size, replace=False, p=probas[:self.mem_cntr])

        else:
            indices = np.random.choice(np.arange(self.mem_cntr), batch_size, replace=False)

        return (
            self.state_memory[indices],
            self.action_memory[indices],
            self.reward_memory[indices],
            self.new_state_memory[indices],
            self.terminal_memory[indices],
            indices
                )


class OnPolicyMemory:
    def __init__(self, batch_size: int):
        self.states = []
        self.next_states = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.probs = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        # np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), \
            np.array(self.actions), \
            np.array(self.probs), \
            np.array(self.vals), \
            np.array(self.rewards), \
            np.array(self.dones), \
            batches

    def store_memory(self, state: np.ndarray, action: int, reward: float,
                     state_: np.ndarray, done: bool, probs: np.ndarray, vals: np.ndarray):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(state_)
        self.probs.append(probs)
        self.values.append(vals)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []
        self.probs = []



