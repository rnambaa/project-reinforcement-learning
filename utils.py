import pandas as pd
import numpy as np
import gymnasium
from gymnasium import spaces
import random
from collections import defaultdict


def preprocess_data(train_data_path, validate_data_path):
    train_data = pd.read_excel(train_data_path)
    validate_data = pd.read_excel(validate_data_path)

    train_data_melted = train_data.melt(
        id_vars="PRICES", var_name="Hour", value_name="Price"
    )
    validate_data_melted = validate_data.melt(
        id_vars="PRICES", var_name="Hour", value_name="Price"
    )

    # Convert hour column to a numeric value
    train_data_melted["Hour"] = (
        train_data_melted["Hour"].str.extract(r"(\d+)").astype(int)
    )
    validate_data_melted["Hour"] = (
        validate_data_melted["Hour"].str.extract(r"(\d+)").astype(int)
    )

    # Sort by date and hour for sequential access
    train_data_melted.sort_values(by=["PRICES", "Hour"], inplace=True)
    validate_data_melted.sort_values(by=["PRICES", "Hour"], inplace=True)

    # rename the PRICES column to Date
    train_data_melted.rename(columns={"PRICES": "Date"}, inplace=True)

    return train_data_melted


class QLearningAgent:
    def __init__(
        self,
        action_space,
        learning_rate=0.01,
        discount_factor=0.9,
        exploration_rate=0.1,
    ):
        self.action_space = action_space
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample()  # Explore action space
        else:
            return np.argmax(self.q_table[state])  # Exploit learned values

    def learn(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error


class SmartGridBatteryEnv(gymnasium.Env):
    def __init__(self, data):
        super(SmartGridBatteryEnv, self).__init__()

        # Constants
        self.max_battery_capacity = 50  # kWh
        self.efficiency = 0.9
        self.max_power = 25  # kW
        self.min_required_battery = 20  # kWh
        self.charge_discharge_limit = 20  # kWh

        # State Space: Battery charge level, Time of day, Electricity prices
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([self.max_battery_capacity, 24, np.inf]),
            dtype=np.float32,
        )

        # Action Space: Charge/Discharge amount (positive for charging, negative for discharging)
        self.action_space = spaces.Box(
            low=-self.max_power, high=self.max_power, shape=(1,), dtype=np.float32
        )

        # Initialize state and data
        self.data = data.reset_index(drop=True)
        self.current_index = 0  # To track the current position in the dataset
        self.state = [
            self.max_battery_capacity / 2,
            self.data.at[self.current_index, "Hour"],
            self.data.at[self.current_index, "Price"],
        ]

    def step(self, action):
        # Extract state components
        battery_level, _, _ = self.state

        # Calculate new battery level considering efficiency
        delta_energy = action * self.efficiency
        new_battery_level = np.clip(
            battery_level + delta_energy, 0, self.max_battery_capacity
        )

        # Update to the next hour in the dataset
        self.current_index = (self.current_index + 1) % len(self.data)
        new_hour = self.data.at[self.current_index, "Hour"]
        new_price = self.data.at[self.current_index, "Price"]

        # Calculate reward (or cost)
        reward = (
            -new_price * delta_energy
        )  # Negative cost for discharging, positive revenue for charging

        # Update state
        self.state = [new_battery_level, new_hour, new_price]

        # Check if the episode is done (e.g., end of the dataset)
        done = self.current_index == 0

        return np.array(self.state), reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_index = 0
        self.state = [
            self.max_battery_capacity / 2,
            self.data.at[self.current_index, "Hour"],
            self.data.at[self.current_index, "Price"],
        ]
        return np.array(self.state)

    def render(self, mode="human"):
        # Optional visualization code here
        pass

    def close(self):
        # Optional cleanup code here
        pass
