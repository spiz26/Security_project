import random
import numpy as np

class QAgent():
    def __init__(self):
        self.q_table = np.zeros((100, 25000, 2, 2, 4))
        self.eps = 0.9

    def select_action(self, s):
        ports, connected_pc, key_sec, web_cre = s
        coin = random.random()
        if coin < self.eps:
            action = random.randint(0,3)
        else:
            action_val = self.q_table[ports, connected_pc, key_sec, web_cre, :]
            action = np.argmax(action_val)
        return action

    def update_table(self, transition):
        s, a, r, s_prime = transition
        ports, connected_pc, key_sec, web_cre = s
        n_ports, n_connected_pc, n_key_sec, n_web_cre = s_prime
        a_prime = self.select_action(s_prime)
        self.q_table[ports, connected_pc, key_sec, web_cre, a] = self.q_table[ports, connected_pc, key_sec, web_cre, a] + 0.1 * (r + np.amax(self.q_table[n_ports, n_connected_pc, n_key_sec, n_web_cre,:]) - self.q_table[ports, connected_pc, key_sec, web_cre, a]) 

    def anneal_eps(self):
        self.eps -= 0.01
        self.eps = max(self.eps, 0.2)