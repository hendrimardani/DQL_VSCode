# gym versi 0.25.0
import torch
import gym
import random
import copy
import torch.nn.functional as F
from torch import nn 
from torch.optim import AdamW
from tqdm import tqdm
from utils import test_agent, plot_stats, seed_everything
from dq_learning import Robot, PreprocessEnv, OtakSaraf
import matplotlib.pyplot as plt

if __name__ == "__main__":
    nama_env = "MountainCar-v0"
    env = gym.make(nama_env)
    state_dims = env.observation_space.shape[0]
    num_actions = env.action_space.n

    robot = Robot(env=nama_env, epsilon=0.2, state_dims=state_dims, num_ations=num_actions)

    preEnv = PreprocessEnv(robot.env_r)

    otak = OtakSaraf(capacity=100000, episodes=15000, alpha=1e-3, env_r=preEnv.reset, 
                    prec_env=preEnv, state_dims=state_dims, num_actions=num_actions)

    # Menampilkan laju mobil belum dilatih
    # robot.cobaan

    # # Training (Latih)
    stats = otak.saraf()
    # test_agent(preEnv, preEnv.reset, otak.policy, episodes=10)

    # ploting hasil training
    # plot_stats(stats)