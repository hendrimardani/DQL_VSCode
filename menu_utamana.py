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
    