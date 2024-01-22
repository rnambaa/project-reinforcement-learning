from TestEnv import Electric_Car
import argparse
import matplotlib.pyplot as plt
import numpy as np
from agents import *
import seaborn as sns
import pandas as pd

AGENTS = [buy5, buy5sell11, random_uniform, random_normal]


# Make the excel file as a command line argument, so that you can do: " python3 main.py --excel_file validate.xlsx "
parser = argparse.ArgumentParser()
parser.add_argument(
    "--excel_file", type=str, default="data/validate.xlsx"
)  # Path to the excel file with the test data
args = parser.parse_args()
# log
log = []


for agent in AGENTS:
    env = Electric_Car(path_to_test_data=args.excel_file)
    observation = env.observation()
    cumulative_reward = 0
    # loop trough validation data
    i = 0
    done = False
    while not done:
        # index of the hour
        i += 1
        # Choose a random action between -1 (full capacity sell) and 1 (full capacity charge)
        # action = env.continuous_action_space.sample()
        # Only choose randomly 1 or -1 or 0
        action = agent(env.hour)

        # Or choose an action based on the observation using your RL agent!:
        # action = RL_agent.act(observation)
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
