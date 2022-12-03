from thorvald_env import ThorvaldEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
import torch as th

# Log directory of the tensorboard files to visualize the training and for the final policy as well
log_dir = "./cnn_policy"
# set headles to False to visualize the policy while training
my_env = ThorvaldEnv(headless=False)


policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[16, dict(pi=[128, 128, 128], vf=[128, 128, 128])]) # Policy params
policy = MlpPolicy
total_timesteps = 1000000

# Saves a checkpoint policy in the same log directory
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="thorvald_policy_checkpoint")
# PPO algorithm params
model = PPO(
    policy,
    my_env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    n_steps=2560,
    batch_size=64,
    learning_rate=0.000125,
    gamma=0.9,
    ent_coef=7.5e-08,
    clip_range=0.3,
    n_epochs=5,
    device="cuda",
    gae_lambda=1.0,
    max_grad_norm=0.9,
    vf_coef=0.95,
    tensorboard_log=log_dir,
)
model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback])

model.save(log_dir + "/thorvald_policy") # Saves the final policy

my_env.close() # Closes the environment