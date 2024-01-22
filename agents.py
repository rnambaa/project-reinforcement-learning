import numpy as np


def buy5(hour: int):
    if hour == 5:
        return 1
    else:
        return 0


def buy5sell11(hour: int):
    if hour == 5:
        return 1
    elif hour == 11:
        return -1
    else:
        return 0


def random_uniform(hour):
    """selects continuous random action between -1 and 1"""
    return np.random.uniform(-1, 1)


def random_normal(hour):
    """selects continuous random action between -1 and 1 from normal distribution"""
    return np.random.normal(0, 1)


class QLearningAgent:
    def __init__(
        self, n_states: int, n_actions: int, alpha: float, gamma: float, epsilon: float
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.q_table = np.zeros((n_states, n_actions))
        self.observation_space = [10, 10, 24, 7, 12, 3]

    def choose_action(self, state: int) -> int:
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.n_actions)  # Explore
        else:
            action = np.argmax(
                self.q_table[state, :]
            )  # Exploit (use the best known action)
        return action

    def learn(self, state: int, action: int, reward: float, next_state: int):
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state, :])
        self.q_table[state, action] += self.alpha * (target - predict)

    def get_state_index(self, state_tuple):
        """Converts a state tuple to a single integer index"""
        state_index = 0
        for i in range(len(state_tuple)):
            state_index += state_tuple[i] * np.prod(self.observation_space[i + 1 :])
        return int(state_index)
