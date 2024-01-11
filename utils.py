import pandas as pd
import numpy as np
import gymnasium
from gymnasium import spaces
import random
from collections import defaultdict


def process_data(path):
    data = pd.read_excel(path)
    # melt the data to have a single column for prices
    data_melted = data.melt(id_vars="PRICES", var_name="Hour", value_name="Price")
    # extract the hour from the hour column
    data_melted["Hour"] = data_melted["Hour"].str.extract(r"(\d+)").astype(int)
    # sort by prices and then by hour
    data_melted.sort_values(by=["PRICES", "Hour"], inplace=True)
    # rename prices to date
    data_melted.rename(columns={"PRICES": "Date"}, inplace=True)
    return data_melted

class QLearningAgent:
    def __init__(self, state_space, action_space, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((state_space.shape[0], action_space.n))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)



class SmartGridBatteryEnv(gymnasium.Env):
    def __init__(self, data):
        super(SmartGridBatteryEnv, self).__init__()

        # Constants
        self.BATTERY_CAPACITY = 50  # kWh
        self.EFFICIENCY = 0.9
        self.MAX_POWER = 25  # kW/h
        self.MIN_BATTERY = 20  # kWh
        self.DAY_USAGE = 20  # kWh

        # variables
        self.car_available = True

        # State Space: Battery charge level, Time of day, Electricity prices
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([self.BATTERY_CAPACITY, 24, np.inf]),
            dtype=np.float32,
        )

        # Define the discrete action space
        self.action_space = spaces.Discrete(11)

        # Initialize state and data
        self.data = data.reset_index(drop=True)
        self.current_index = 0  # To track the current position in the dataset

        # reset the state to the initial state
        self.state = self.reset()

    @staticmethod
    def adjust_battery_for_availability(self, battery_level: float, hour: int):
        """Adjust the battery level if the car is not available (being used)) between 8am and 6pm"""
        if hour == 18 and not self.car_available:
            battery_level -= self.DAY_USAGE
        return battery_level

    def update_state(self, battery_level: float):
        self.current_index = (self.current_index + 1) % len(self.data)
        new_hour = self.data.at[self.current_index, "Hour"]
        new_price = self.data.at[self.current_index, "Price"]
        # Update state
        self.state = [battery_level, new_hour, new_price]

    def get_power_value(self, action_index):
        """Converts an action index to a power value. Power value is range between -25 and 25 in steps of 5"""
        if action_index < 0 or action_index > 10:
            raise ValueError("Invalid action index. Must be between 0 and 10.")
        return (action_index - 5) * 5

    def step(self, action: int):
        action = self.get_power_value(action)
        # Extract state components
        battery_level, hour, price = self.state
        reward = 0

        # update the car availability randomly every day
        if hour == 1:
            self.car_available = random.choice([True, False])

        # Adjust battery level if the car is not available
        battery_level = self.adjust_battery_for_availability(self, battery_level, hour)

        # Calculate new battery level if car is available or if it is outside the 8am-6pm window
        if self.car_available or (hour < 8 or hour > 18):
            # Calculate new battery level considering efficiency
            delta_energy = action * self.EFFICIENCY
            if delta_energy > 0:
                # charging
                max_possible_charging_amount = self.MAX_POWER - battery_level
                charging_amount = min(max_possible_charging_amount, delta_energy)
                battery_level += charging_amount
                # pay the price to charge the battery
                reward -= 2 * price * charging_amount / self.EFFICIENCY
            else:
                # discharging
                max_possible_discharging_amount = -battery_level
                discharging_amount = max(max_possible_discharging_amount, delta_energy)
                battery_level += discharging_amount
                # get paid the price to discharge the battery
                reward += price * discharging_amount / self.EFFICIENCY

        # ensure the battery level is at least 20kWh at 7am
        if hour == 7 and battery_level < 20:
            charge_needed = 20 - battery_level
            # pay the difference in price to charge the battery
            reward -= 2 * price * charge_needed / self.EFFICIENCY

            battery_level = 20

        # update the state
        self.update_state(battery_level)

        # Check if the episode is done (e.g., end of the dataset)
        done = self.current_index == 0

        # return the new state, reward, done, and any additional information
        return np.array(self.state), reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_index = 0
        # state = [battery_level, hour, price]
        self.state = [
            self.BATTERY_CAPACITY / 2,
            self.data.at[self.current_index, "Hour"],
            self.data.at[self.current_index, "Price"],
        ]
        return np.array(self.state)

    def render(self, mode="human"):
        # Print the current state
        print(f"Current state: {self.state}")

    def close(self):
        # Optional cleanup code here
        pass
