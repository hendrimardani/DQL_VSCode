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
import matplotlib.pyplot as plt


class Robot:
    def __init__(self, env):
        self.env = env

    @property
    def cobaan(self):
        env = gym.make(self.env)
        env.reset()
        tahan_layar = 100
        actions_ran = env.action_space.sample()

        for _ in range(tahan_layar):
            env.step(actions_ran)
            env.render()

        return env
    def informasi(self):
        env = self.cobaan
        state_dims = env.observation_space.shape[0]
        num_actions = env.action_space.n
        return f"Jumlah state = {state_dims}\n Jumlah Actions = {num_actions}"
    
    @property
    def env_(self):
        env = gym.make(self.env)
        env.reset()

        return env
        
class PreprocessEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self):
        state = self.env.reset()
        
        return torch.from_numpy(state).unsqueeze(dim=0).float()
    
    def step(self, action):
        action = action.item()
        next_state, reward, done, info = self.env.step(action)
        next_state = torch.from_numpy(next_state).unsqueeze(dim=0).float()
        reward = torch.tensor(reward).view(1, -1).float()
        done = torch.tensor(done).view(1, -1)

        return next_state, reward, done, info

robot = Robot(env="MountainCar-v0")

env = PreprocessEnv(env=robot.env_)
action = torch.tensor(0)
# next_state, reward, done, _ = env.step(action)
print(env.step(action))

# hasil = Robot(env="MountainCar-v0")
# print(hasil.informasi())