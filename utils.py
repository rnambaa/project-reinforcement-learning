import pandas as pd
import numpy as np
import gymnasium
from gymnasium import spaces
import seaborn as sns
import random
import matplotlib.pyplot as plt



def plot_log(log_df, hours=48):
    # plot the log: price, action, battery level, reward, car available
    fig, axes = plt.subplots(6, 1, figsize=(15, 15), sharex=True)
    # select the first 48 hours
    log_df = log_df.iloc[:hours]

    sns.lineplot(ax=axes[0], data=log_df, x=log_df.index, y="Price")
    sns.lineplot(ax=axes[1], data=log_df, x=log_df.index, y="Action")
    sns.lineplot(ax=axes[2], data=log_df, x=log_df.index, y="Battery Level")
    sns.lineplot(ax=axes[3], data=log_df, x=log_df.index, y="Reward")
    sns.lineplot(ax=axes[4], data=log_df, x=log_df.index, y="Hour")
    sns.lineplot(ax=axes[5], data=log_df, x=log_df.index, y="Car Available")

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

    # add column for day of the week
    data_melted["Day of Week"] = data_melted["Date"].dt.dayofweek

    # bin prices
    data_melted = bin_prices_with_index(data_melted, 'Price', 500, 10)

    return data_melted

def bin_prices_with_index(data: pd.DataFrame, price_column: str, high_price_threshold: int, step_size: int) -> pd.DataFrame:

    # Constants
    HIGH_PRICE_BIN_INDEX = high_price_threshold // step_size

    # Function to determine the bin index for each price
    def determine_bin_index(price: float) -> int:
        if price > high_price_threshold:
            return HIGH_PRICE_BIN_INDEX
        else:
            return int(price // step_size)

    # Apply the binning function to the price column
    data['Price_Bin_Index'] = data[price_column].apply(determine_bin_index)
    return data


class QLearningAgent:
    def __init__(self, state_space, action_space, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # self.q_table = np.zeros((state_space, action_space))
        self.q_table = {}  # Nested dictionaries for Q-values

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return np.argmax(self.q_table[state])


    def get_q_value(self, state, action):
        # Returns the Q-value for a given state-action pair
        return self.q_table.get(state, {}).get(action, 0)


    # def update_q_table(self, state, action, reward, next_state):
    #     best_next_action = np.argmax(self.q_table[next_state])
    #     td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
    #     td_error = td_target - self.q_table[state, action]
    #     self.q_table[state, action] += self.alpha * td_error


    def update_q_table(self, state, action, reward, next_state):
        # Update Q-table with new knowledge
        next_max = max(self.q_table.get(next_state, {}).values(), default=0)
        self.q_table.setdefault(state, {})[action] = (1 - self.alpha) * self.get_q_value(state, action) + \
                                                      self.alpha * (reward + self.gamma * next_max)


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
        self.battery_step_size = 5
        self.log = []

        # number of states
        # self.num_battery_levels = int(50 / self.battery_step_size) # this is wrong, because of the discount factor..
        self.num_battery_levels = 100 # adhoc
        self.num_hours = 24
        self.num_price_bins = 51
        self.num_days_of_week = 7


        # # Discrete State Space
        # self.observation_space = spaces.Tuple([
        #     spaces.Discrete(int(50 / self.battery_step_size)),  # Battery charge level
        #     spaces.Discrete(24),                                # Time of day
        #     spaces.Discrete(51),                                # Electricity prices
        #     spaces.Discrete(7)                                  # day of the week
        # ])

        # Discrete State Space - integer representation (for Q-learning)
        # NOTE: this is only used in the q-learning training loop
        self.observation_space = spaces.Discrete(
            int(self.num_battery_levels * self.num_hours * self.num_price_bins * self.num_days_of_week)
            )




        # Define the discrete action space (Battery charge level)
        self.action_space = spaces.Discrete(int((50 / self.battery_step_size) + 1)) # +1 since indexed at 0

        # Initialize state and data
        self.data = data.reset_index(drop=True)
        self.current_index = 0  # To track the current position in the dataset

        # reset the state to the initial state
        self.state = self.reset()

    @staticmethod
    def use_car(self, battery_level: float, hour: int):
        """Adjust the battery level if the car is not available (being used)) between 8am and 6pm"""
        if hour == 18 and not self.car_available:
            battery_level -= self.DAY_USAGE
        return battery_level


    def update_state(self, battery_level: float):
        self.current_index = (self.current_index + 1) % len(self.data)
        new_hour = self.data.at[self.current_index, "Hour"]
        new_price_bin = self.data.at[self.current_index, "Price_Bin_Index"]
        new_day = self.data.at[self.current_index, "Day of Week"]
        self.price = self.data.at[self.current_index, "Price"]
        # Update state
        self.state = [battery_level, new_hour, new_price_bin , new_day]


    def get_state_index(self, state_array):
        """
        Converts a state tuple to a unique integer index.
        :param state_array: An array containing (battery_level, hour, price_bin, day_of_week)
        :return: A unique integer representing the state.
        """
        battery_level, hour, price_bin, day_of_week = state_array

        # Define the range for each component of the state
        num_battery_levels = self.num_battery_levels
        num_hours = self.num_hours
        num_price_bins = self.num_price_bins
        num_days = self.num_days_of_week

        # Calculate the unique index
        index = (battery_level * num_hours * num_price_bins * num_days +
                hour * num_price_bins * num_days +
                price_bin * num_days +
                day_of_week)

        return int(index)


    def get_power_value(self, action_index, battery_step_size):
        """Converts an action index to a power value. Power value is range between -25 and 25 in for a given battery step size."""
        max_action_index = (50 / battery_step_size)
        if action_index < 0 or action_index > (max_action_index):
            raise ValueError("Invalid action index. Must be between 0 and 10.")

        return (action_index - max_action_index / 2) * battery_step_size



    def step(self, action: int):
        action = self.get_power_value(action, self.battery_step_size)
        # Extract state components
        battery_level, hour, price_index, day_of_week = self.state
        reward = 0
        price = self.price

        # update the car availability randomly every day
        if hour == 1:
            self.car_available = random.choice([True, False])

        # Adjust battery level if the car is not available
        battery_level = self.use_car(self, battery_level, hour)

        # Calculate new battery level if car is available or if it is outside the 8am-6pm window
        if self.car_available or (hour < 8 or hour > 18):
            # Calculate new battery level considering efficiency
            delta_energy = action * self.EFFICIENCY
            if delta_energy > 0:
                # charging
                max_possible_charging_amount = self.BATTERY_CAPACITY - battery_level
                charging_amount = min(max_possible_charging_amount, delta_energy)
                battery_level += charging_amount
                # pay the price to charge the battery
                reward -= 2 * (price / 1000) * charging_amount / self.EFFICIENCY
            else:
                # discharging
                max_possible_discharging_amount = -battery_level
                discharging_amount = max(max_possible_discharging_amount, delta_energy)
                battery_level += discharging_amount
                # make discharge amount positive
                discharging_amount = abs(discharging_amount)
                # get paid the price to discharge the battery

                reward += (price / 1000) * discharging_amount / self.EFFICIENCY




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

        # log action, battery level, hour, price, reward as dict
        self.log.append({
            "Action": action,
            "Battery Level": battery_level,
            "Hour": hour,
            "Price": price,
            "Reward": reward,
            "Car Available": self.car_available,
        })

        # return the new state, reward, done, and any additional information
        return np.array(self.state), reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_index = 0
        self.state = [
            self.BATTERY_CAPACITY / 2,
            self.data.at[self.current_index, "Hour"],
            self.data.at[self.current_index, "Price_Bin_Index"],
            self.data.at[self.current_index, "Day of Week"]
        ]
        self.price = self.data.at[self.current_index, "Price"]
        # add the initial state to the log
        self.log.append({
            "Action": 0,
            "Battery Level": self.state[0],
            "Hour": self.state[1],
            "Price": self.price,
            "Reward": 0
        })
        return np.array(self.state)

    def render(self, mode="human"):
        # Print the current state
        print(f"Current state: {self.state}")

    def close(self):
        # Optional cleanup code here
        pass
