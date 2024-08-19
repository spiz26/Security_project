import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.optim.lr_scheduler import StepLR

#Hyperparameters
learning_rate = 0.001
gamma = 0.98
n_rollout = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, num_states, num_actions):
        super(ActorCritic, self).__init__()
        self.data = []
        
        hidden1 = 64
        hidden2 = 64
        self.fc1 = nn.Linear(num_states, hidden1)
        #self.fc2 = nn.Linear(32, 64)
        self.fc_pi = nn.Linear(hidden1, num_actions)
        self.fc_v = nn.Linear(hidden1, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        #self.scheduler = StepLR(self.optimizer, step_size=
                                 
    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        prob = F.softmax(self.fc_pi(x), dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        v = self.fc_v(x)
        return v
    
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_list, a_list, r_list, s_prime_list, done_list = [], [], [], [], []
        for transition in self.data:
            s,a,r,s_prime,done = transition
            s_list.append(s)
            a_list.append([a])
            r_list.append([r/100.])
            s_prime_list.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_list.append([done_mask])
        
        s_batch = torch.tensor(np.array(s_list), dtype=torch.float).to(device)
        a_batch = torch.tensor(np.array(a_list)).to(device)
        r_batch = torch.tensor(np.array(r_list), dtype=torch.float).to(device)
        s_prime_batch = torch.tensor(np.array(s_prime_list), dtype=torch.float).to(device)
        done_batch = torch.tensor(np.array(done_list), dtype=torch.float).to(device)
                                                              
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch
    
    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)
        
        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1,a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        
        return self.optimizer.param_groups[0]["lr"]