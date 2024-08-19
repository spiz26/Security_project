import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#Hyperparameters
learning_rate = 0.001
gamma = 0.98
n_rollout = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NActorCritic(nn.Module):
    def __init__(self, num_states, num_actions):
        super(NActorCritic, self).__init__()
        self.data = []
        
        hidden1 = 64
        hidden2 = 64
        self.fc1 = nn.Linear(num_states, hidden1)
        #self.fc2 = nn.Linear(32, 64)
        self.fc_pi = nn.Linear(hidden1, num_actions)
        self.fc_v = nn.Linear(hidden1, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
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
        a2_list, r2_list, s_prime2_list, done2_list = [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done, a2, r2, s_prime2, done2 = transition
            s_list.append(s)
            a_list.append([a])
            r_list.append([r/100.])
            s_prime_list.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_list.append([done_mask])
            
            a2_list.append([a2])
            r2_list.append([r2/100.])
            s_prime2_list.append(s_prime2)
            done_mask2 = 0.0 if done2 else 1.0
            done2_list.append([done_mask2])
            
        s_batch = torch.tensor(s_list, dtype=torch.float).to(device)
        a_batch = torch.tensor(a_list).to(device)
        r_batch = torch.tensor(r_list, dtype=torch.float).to(device)
        s_prime_batch = torch.tensor(s_prime_list, dtype=torch.float).to(device)
        done_batch = torch.tensor(done_list, dtype=torch.float).to(device)
        
        a2_batch = torch.tensor(a2_list).to(device)
        r2_batch = torch.tensor(r2_list, dtype=torch.float).to(device)
        s_prime2_batch = torch.tensor(s_prime2_list, dtype=torch.float).to(device)
        done2_batch = torch.tensor(done2_list, dtype=torch.float).to(device)
        
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch, a2_batch, r2_batch, s_prime2_batch, done2_batch
    
    def train_net(self):
        s, a, r, s_prime, done, a2, r2, s_prime2, done2 = self.make_batch()
        td_target = (r) + (gamma * r2 * done) + (gamma ** 2 * self.v(s_prime2) * done2)
        delta = td_target - self.v(s)
        
        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1, a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()