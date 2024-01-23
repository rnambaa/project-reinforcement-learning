import os
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common import results_plotter

LOG_DIR = "/tmp/gym/"
MODEL_PATH = "models/PPO"
TENSORBOARD_LOG = "tensorboard_log/"
N_EVAL_EPISODES = 2
EVAL_FREQ: int = 10
TOTAL_TIMESTEPS: int = 10

# Assuming Electric_Car is a gym.Env
from TestEnv import Electric_Car

# Create the environment
env = make_vec_env(lambda: Monitor(Electric_Car(), LOG_DIR), n_envs=1)

# Instantiate the agent with TensorBoard logging
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=TENSORBOARD_LOG, device="mps")

# Create evaluation callback
eval_callback = EvalCallback(
    env,
    best_model_save_path=MODEL_PATH,
    log_path=LOG_DIR,
    eval_freq=EVAL_FREQ,
    deterministic=True,
    render=False,
)

# Train the agent
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)

# Save the agent
model.save(MODEL_PATH)
print("Model saved")

# Evaluate the policy
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=N_EVAL_EPISODES)
print(f"Mean reward: {mean_reward}, Std Reward: {std_reward}")

# Plot results
plot_results(
    [LOG_DIR], TOTAL_TIMESTEPS, results_plotter.X_TIMESTEPS, "PPO Electric Car"
)
