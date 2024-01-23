from TestEnv import Electric_Car
import numpy as np
from agents import QLearningAgent
import pandas as pd


def discretize_state(state: np.ndarray) -> np.ndarray:
    BATTERY_LEVEL_BINS = 10  # Adjust as needed
    PRICE_BINS = 10  # Adjust as needed
    battery_level, price, hour, day_of_week, day_of_year, month, year = state

    # Discretize battery level
    discretized_battery_level = np.digitize(
        battery_level, bins=np.linspace(0, 50, BATTERY_LEVEL_BINS)
    )  # Assuming battery level ranges from 0 to 50

    # Discretize price
    discretized_price = np.digitize(
        price, bins=np.linspace(0.01, 3000, PRICE_BINS)
    )  # min_price and max_price should be set based on data

    # Combine into a single array
    discretized_state = np.array(
        [
            discretized_battery_level,
            discretized_price,
            hour,
            day_of_week,
            day_of_year,
            month,
            year,
        ]
    )
    # calculate state space size

    return discretized_state


# Define parameters
n_states = (
    10 * 10 * 24 * 7 * 12 * 3
)  # Define the number of states (after discretization if necessary)
n_actions = 10  # Define the number of actions
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.1  # exploration rate

agent = QLearningAgent(n_states, n_actions, alpha, gamma, epsilon)

# length of dataset

# Create a list to log the data
log_data = []


# Learning loop
n_episodes = 1000  # Set the number of episodes
for episode in range(n_episodes):
    env = Electric_Car(path_to_test_data="data/train.xlsx")
    n = len(env.test_data)
    done = False
    cumulative_reward = 0

    while not done:
        state = env.observation()
        # discretize state
        state_discrete = discretize_state(state)
        # convert state to index
        state_index = agent.get_state_index(state_discrete)

        action = agent.choose_action(state_index)
        action_continuous = action / 10 - 1

        if done or env.counter == n - 1:
            break

        next_state, reward, done, truncated, _ = env.step(action_continuous)
        # discretize next_state
        next_state_discrete = discretize_state(next_state)
        # convert next_state to index
        next_state_index = agent.get_state_index(next_state_discrete)

        agent.learn(state_index, action, reward, next_state_index)
        state = next_state
        cumulative_reward += reward

    log_data.append(
        {
            "Episode": episode,
            "Reward": cumulative_reward,
        }
    )


# Convert the list to a pandas DataFrame
log_data_df = pd.DataFrame(log_data)

# plot cumulative reward over time using seaborn
import seaborn as sns
import matplotlib.pyplot as plt

sns.lineplot(data=log_data_df, x="Episode", y="Reward")
plt.show()

# save plot
plt.savefig("plots/qlearning.png")
