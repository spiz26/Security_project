import os
import gym
import collections
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from security_simulation import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tau = 0.001

class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
        
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_list, a_list, r_list, s_prime_list, done_mask_list = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_list.append(s)
            a_list.append([a])
            r_list.append([r])
            s_prime_list.append(s_prime)
            done_mask_list.append([done_mask])
        
        s_list = torch.tensor(s_list, dtype=torch.double).to(device)
        a_list = torch.tensor(a_list).to(device)
        r_list = torch.tensor(r_list).to(device)
        s_prime_list = torch.tensor(s_prime_list, dtype=torch.double).to(device)
        done_mask_list = torch.tensor(done_mask_list).to(device)
        
        return s_list, a_list, r_list, s_prime_list, done_mask_list
    
    def size(self):
        return len(self.buffer)
    
class Qnet(nn.Module):
    def __init__(self, num_states, num_actions):
        super(Qnet, self).__init__()
        
        self.num_states = num_states
        self.num_actions = num_actions
        
        self.fc1 = nn.Linear(self.num_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.num_actions)
        self.double()
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = np.random.random()
        
        if coin < epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            return out.argmax().item()
        
def fitting_model(q, q_target, memory, optimizer):
    s, a, r, s_prime, done_mask = memory.sample(batch_size)
        
    q_a = q(s).gather(1, a)
    max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
    target = r + gamma * max_q_prime * done_mask
    loss = F.mse_loss(q_a, target)  # smooth_l1_loss
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        
    soft_update(q, q_target, tau)
    
def soft_update(local_model, target_model, tau):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

def train(args, num_pc, port_size):
    lr, gamma, buffer_limit, batch_size, episodes, init_eps = args
    
    num_state = 4
    num_action = 4
    
    q = Qnet(num_state, num_action).to(device)
    q_target = Qnet(num_state, num_action).to(device)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer(buffer_limit)
    
    score_list = []
    success_rate_list = []
    try_list = []
    suc_act = success_action()
    _try = 0
    _success = 0
    
    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=lr)
    epsilon = init_eps
    trans_limit = 1000
    step = 0
    
    for n_epi in range(1, episodes+1):
        epsilon = max(0.01, epsilon*0.99)
        seed = int(str(t0)[11:14])
        env = security(num_pc=num_pc, seed=seed, port_size=port_size)
        s = env.reset()
        done = False
        
        for i in range(trans_limit):
            
            a = q.sample_action(torch.from_numpy(s).double().to(device), epsilon)
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            
            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime
            score += r
            step += 1
            if info[2]:
                _try += 1
                _success += 1
            else:
                _try += 1
        
        
            if memory.size() > 5000:
                if step % 4 == 0:
                    fitting_model(q, q_target, memory, optimizer)
            if done:
                break
        
        suc_act.put(select_action(a))
        if n_epi % print_interval == 0:
            print(f"\rn_episode:{n_epi}, score:{round(score/print_interval, 2)}, n_buffer:{memory.size()}, eps:{round(epsilon*100, 2)}") 
            score_list.append(score / print_interval)
            success_rate_list.append(round(_success/_try, 2))
            try_list.append(_try)
            score = 0.0
            _try = 0
            _success = 0
            suc_act.reset()
                
        if n_epi != episodes:
            print(f"\r{n_epi % print_interval} / {print_interval}", end="")

    return q, score_list, success_rate_list, try_list, suc_act.load_list()

