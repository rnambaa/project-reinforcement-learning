from TestEnv import Electric_Car
import argparse
import matplotlib.pyplot as plt
import numpy as np
from agents import *
import seaborn as sns
import pandas as pd

EPISODES = 10


# Make the excel file as a command line argument, so that you can do: " python3 main.py --excel_file validate.xlsx "
parser = argparse.ArgumentParser()
parser.add_argument(
    "--excel_file", type=str, default="data/validate.xlsx"
)  # Path to the excel file with the test data
args = parser.parse_args()
# log
log = []

# add agent class
class QLearningAgent:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.1,
        discount_factor=0.99,
        exploration_rate=1.0,
        exploration_decay_rate=0.995,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.q_table = np.zeros((state_size, action_size))
        # tracking
        self.visited_pairs = set()


    # tracking
    def has_been_visited(self, state, action):
        """Check if a state-action pair has been visited."""
        return (state, action) in self.visited_pairs

    def mark_visited(self, state, action):
        """Mark a state-action pair as visited."""
        self.visited_pairs.add((state, action))

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])

        # Update the Q-Value
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (
            reward + self.discount_factor * next_max
        )
        self.q_table[state, action] = new_value

        # Decay exploration rate
        if done:
            self.exploration_rate *= self.exploration_decay_rate

    def bin_state(self, state, min_value, max_value, num_bins):
        '''
        discretized a continuous state value by putting it in a bin and returning its index.
        '''
        bin_width = (max_value - min_value) / num_bins
        bin_index = int((state - min_value) / bin_width)
        return bin_index


    # add action/state space discretization function
    def get_state_index(self, observation):

        # range of each component
        NUM_BATTERY_LEVEL = 100 # TODO: bins
        NUM_PRICE_BINS = ... # TODO: BINS
        NUM_HOURS = 24
        NUM_DAYS_OF_WEEK = 7
        NUM_DAYS_OF_YEAR = 365
        NUM_MONTHS = 12
        # IGNORE YEAR




                battery_level,
                price,
                int(hour),
                int(day_of_week),
                int(day_of_year),
                int(month),
                int(year),



        # function for converting price & battery level into bins
        # the value of the price & battery bin will change depending on the observation


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

# TODO
state_size = ...
action_size = ...
print(f"state size {state_size}")
print(f"action size {action_size}\n")

# Initialize Q-Learning agent
agent = QLearningAgent(state_size, action_size)


for episode in EPISODES:
    env = Electric_Car(path_to_test_data=args.excel_file)
    observation = env.observation()
    cumulative_reward = 0
    # loop trough validation data
    i = 0
    done = False
    while not done:
        # index of the hour
        i += 1

        # TODO: get the state index here

        # agent chooses action
        action = agent.choose_action(...) # TODO: give discretized observation

        # TODO: discretize the action index here.


        # The observation is the tuple: [battery_level, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        observation = next_observation

        # add reward to cumulative reward
        cumulative_reward += reward

        # add to log as dict
        log.append(
            {
                "agent": agent.__name__,
                "hour": i,
                "action": action,
                "battery_level": observation[0],
                "price": observation[1],
                "hour_of_day": observation[2],
                "day_of_week": observation[3],
                "day_of_year": observation[4],
                "month_of_year": observation[5],
                "year": observation[6],
                "reward": reward,
                "cumulative_reward": cumulative_reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
            }
        )


# convert log to dataframe
log = pd.DataFrame(log)

# plot cumulative reward over time for each agent
sns.lineplot(data=log, x="hour", y="cumulative_reward", hue="agent")
plt.show()
