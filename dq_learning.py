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
    def __init__(self, env, epsilon, state_dims, num_ations):
        self.env = env
        self.epsilon = epsilon
        self.state_dims = state_dims
        self.num_actions = num_ations

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
    
    @property
    def jum_statedims_numactions(self):
        env = self.cobaan
        state_dims = env.self.state_dims
        num_actions = env.self.num_actions

        return state_dims, num_actions

    @property
    def informasi(self):
        state_dims, num_actions = self.jum_statedims_numactions

        return f"Jumlah state = {state_dims}\n Jumlah Actions = {num_actions}"
    
    @property
    def env_r(self):
        env = gym.make(self.env)
        env.reset()

        return env

class PreprocessEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    @property
    def reset(self):
        state = self.env.reset()
        
        return torch.from_numpy(state).unsqueeze(dim=0).float()
    
    def step(self, action):
        action = torch.tensor(0)
        action = action.item()
        next_state, reward, done, info = self.env.step(action)
        next_state = torch.from_numpy(next_state).unsqueeze(dim=0).float()
        reward = torch.tensor(reward).view(1, -1).float()
        done = torch.tensor(done).view(1, -1)

        return next_state, reward, done, info

class ReplayMemory:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def insert(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):    
        assert self.can_sample(batch_size)
        batch = random.sample(self.memory, batch_size)
        batch = zip(*batch)
        
        return [torch.cat(items) for items in batch]

    def can_sample(self,batch_size):
        return len(self.memory) >= batch_size * 10
    
    def __len__(self):
        return len(self.memory)


class OtakSaraf(ReplayMemory):
    def __init__(self, capacity=100000, episodes=1500, alpha=1e-3, batch_size=32, \
                 gamma=0.99, epsilon=0.2, env_r=None, prec_env=None, state_dims=None, \
                 num_actions=None):
        super(OtakSaraf, self).__init__()
        self.capacity = capacity
        self.env_r = env_r
        self.epsilon = epsilon
        self.episodes = episodes
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.q_network, _ = self.otak
        _, self.target_q_network = self.otak
        self.alpha = alpha
        self.batch_size = batch_size
        self.gamma = gamma
        self.prec_env = prec_env

    @property
    def otak(self):
        self.q_network = nn.Sequential(
            nn.Linear(self.state_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions)
        )
        self.target_q_network = copy.deepcopy(self.q_network).eval()

        return self.q_network, self.target_q_network
    
    def policy(self):
        state = self.env_r

        if torch.rand(1) < self.epsilon:
            return torch.randint(self.num_actions, (1, 1))
        else:
            av = self.q_network(state).detach()

            return torch.argmax(av, dim=-1, keepdim=True)
        
    def saraf(self):
        optim = AdamW(self.q_network.parameters(), lr=self.alpha)
        stats = {"MSE Loss": [], "Hasilnya":[]}

        for episode in tqdm(range(1, self.episodes + 1)):
            state = self.env_r
            done = False
            ep_return = 0

            while not done:
                action = self.policy()
                next_state, reward, done, _ = self.prec_env.step(action)
                self.insert([state, action, reward, done, next_state])

                if self.can_sample(self.batch_size):
                    state_b, action_b, reward_b, done_b, next_state_b = self.sample(self.batch_size)
                    qsa_b = self.q_network(state_b).gather(1, action_b)

                    next_qsa_b = self.target_q_network(next_state_b)
                    next_qsa_b = torch.max(next_qsa_b, dim=-1, keepdim=True)[0]

                    target_b = reward_b + ~done_b * self.gamma * next_qsa_b
                    loss = F.mse_loss(qsa_b, target_b)

                    self.q_network.zero_grad()
                    loss.backward()
                    optim.step()

                    stats["MSE Loss"].append(loss.item())

                state = next_state
                ep_return += reward.item()

            stats["Hasilnya"].append(ep_return)

            if episode % 10 == 0:
                self.target_q_network.load_state_dict(self.q_network.state_dict())

        return stats

nama_env = "MountainCar-v0"
env = gym.make(nama_env)
state_dims = env.observation_space.shape[0]
num_actions = env.action_space.n

robot = Robot(env=nama_env, epsilon=0.2, state_dims=state_dims, num_ations=num_actions)

preEnv = PreprocessEnv(robot.env_r)

otak = OtakSaraf(capacity=100000, episodes=1500, alpha=1e-3, env_r=preEnv.reset, 
                 prec_env=preEnv, state_dims=state_dims, num_actions=num_actions)

stats = otak.saraf()
stats