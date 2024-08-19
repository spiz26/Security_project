import numpy as np
from time import time as t
from typing import List, Dict, Union, Tuple

t0 = t()
#seed = int(str(t0)[11:14])
seed = 420
np.random.seed(seed)
port_max = 25000

class PC:
    vulnerability_list = ["A", "B", "C", "D", "E", "F", "G"]
    whole_nodes = None
    def __init__(self, name: str, ip: str, permission_port: list, connected_nodes: list):
        """
        permission_port : open port.
        ispw_port : open port with password.
        connected_nodes : connected_nodes(PC object)
        connected_nodes_id : connected_nodes(ip : str)
        """
        self.name = name
        self.ip = ip
        self.permission_port = permission_port
        self.connected_nodes = connected_nodes
        self.connected_nodes_id = []
        self.vulnerability = PC.vulnerability_list[int(np.random.randint(low=0, high=len(PC.vulnerability_list), size=1))] 
        self.credential = str(np.random.rand(1))[3:10]
        self.other_credential = {}
        self.Administrator_privileges = True
        self.infected = False
        self.mail_box = []

        """state1"""
        self.ispw_port = permission_port[np.random.choice(len(permission_port), int(len(permission_port)*np.random.rand(1)), replace=False)]

        """state2"""
        self.permission_pc = connected_nodes[np.random.choice(len(connected_nodes), int(len(connected_nodes)*0.7), replace=False)]

        """state3"""
        self.key_security = 1 if np.random.rand(1) > 0.1 else 0

        """state4"""
        self.web_cre = 1 if np.random.rand(1) > 0.9 else 0

    def env_setting(self):
        """Initialize environment"""
        for p in self.permission_pc:
            self.other_credential[PC.whole_nodes[p].ip] = PC.whole_nodes[p].credential 
        for c in self.connected_nodes:
            self.connected_nodes_id.append(PC.whole_nodes[c].ip)
            
    def export_credential(self, vulnerability: str) -> Tuple[bool, list, dict]:
        """If a vulnerability is breached, return the credentials"""
        if self.vulnerability == vulnerability:
            return True, self.connected_nodes_id, self.other_credential
        else:
            return False, self.connected_nodes_id, None
    
    def allow_access(self, port: int) -> Tuple[bool, Union[bool, None]]:
        """If somebody accesses open port without password, allow accessing"""
        if port in self.permission_port and port not in self.ispw_port:
            return True, self.Administrator_privileges
        else:
            return False, None
        
    def allow_admin(self, ip: str=None, credential: str=None) -> Tuple[bool, Union[bool, None]]:
        """If credential matched, return administrator's privileges"""
        if ip in self.other_credential.keys() or credential == self.credential:
            return True, self.Administrator_privileges
        else:
            return False, None
        
    def receive_mail(self, mail: str) -> bool:
        """receive virus mail"""
        rand_num = np.random.rand(1)
        probability = np.random.randint(low=90, high=100) * 0.01
        if rand_num > probability:
            return False
        else:
            self.mail_box.append(mail)
            return True
    
    def open_mail(self) -> Tuple[bool, Union[bool, None]]:
        """open virus mail(execution)"""
        if len(self.mail_box) != 0:
            rand_num = np.random.rand(1)
            probability = np.random.randint(low=80, high=90) * 0.01
            if rand_num > probability:
                return False, None
            else:
                self.infected = True
                return True, self.Administrator_privileges
        else:
            return False, None

    def key(self) -> Tuple[bool, Union[bool, None]]:
        if self.key_security:
            return False, None
        else:
            return True, self.Administrator_privileges

    def web_credential(self) -> Tuple[bool, Union[bool, None]] :
        if self.web_cre:
            return True, self.Administrator_privileges
        else:
            return False, None
    
    def __str__(self):
        return (f"""name : {self.name}
ip : {self.ip}
port : {sorted(self.permission_port)}
ispw : {sorted(self.ispw_port)}
connected nodes : {sorted(self.connected_nodes)}
permission pc : {self.permission_pc}
vulnerability : {self.vulnerability}
credential : {self.credential}""")

class attacker:
    """
    Actions (4) : try_port_access, ip spoofing, key_logging, access_web
    """
    def __init__(self):
        self.discover_nodes = []
        self.port = {}
        self.conn_nodes = {}
        self.conn_credential = {}
        self.coll_credential = {}
        self.pc_admin = {} #The final goal
        self.key_security = {}
        self.web_cre = {}
        self.mail = "hello!"
        
    def exploit_attack(self, target: PC, vulnerability: str) -> Union[list, str]:
        success, discover_nodes, credential = target.export_credential(vulnerability)
        if target.ip not in self.discover_nodes:
            self.discover_nodes.append(target.ip)
        
        if target.ip not in self.conn_nodes.keys():
            self.conn_nodes[target.ip] = []
        
        for n in discover_nodes:
            if n not in self.discover_nodes:
                self.discover_nodes.append(n)
            if n not in self.conn_nodes[target.ip]:
                self.conn_nodes[target.ip].append(n)
                
        for n in self.discover_nodes:
            if n not in self.key_security.keys():
                self.key_security[target.ip] = target.key_security
            if n not in self.web_cre.keys():
                self.web_cre[target.ip] = target.web_cre
            
        if credential is None:
            pass
        
        else:
            self.conn_credential[target.ip] = credential
            for pc in credential.keys():
                if pc not in self.coll_credential.keys():
                    self.coll_credential[pc] = credential[pc]
    
        self.list_sorting()
        if success:
            return self.conn_credential[target.ip]
        return self.discover_nodes
    
    def exploit_credential(self, target: PC, credential: str):
        """environment attack1"""
        success, self.pc_admin[target.ip] = target.allow_admin(credential=credential)
        return success
    
    def portscan_attack(self, target: PC) -> list:
        """environment attack2"""
        self.port[target.ip] = []
        for p in range(port_max):
            if p in target.permission_port:
                self.port[target.ip].append(p)
        return self.port[target.ip]
    
    def spear_phishing(self, target: PC) -> bool:
        """environment attack3"""
        success = target.receive_mail(self.mail)
        return success
    
    def try_port_access(self, target: PC) -> bool:
        """trainable attack1 - action1"""
        select_port_idx = np.random.randint(low=0, high=len(self.port[target.ip]))
        select_port = self.port[target.ip][select_port_idx]
        success, self.pc_admin[target.ip] = target.allow_access(select_port)
        return success
    
    def spoofing_login(self, target: PC, ip: str) -> bool:
        """trainable attack2 - action2"""
        success, self.pc_admin[target.ip] = target.allow_admin(ip=ip)
        return success

    def key_logging(self, target: PC) -> bool:
        """trainable attack3 - action3"""
        success, self.pc_admin[target.ip] = target.key()
        return success
    
    def access_web(self, target: PC) -> bool:
        """trainable attack4 - action4"""
        success, self.pc_admin[target.ip] = target.web_credential()
        return success
    
    def list_sorting(self):
        self.discover_nodes = sorted(self.discover_nodes)
        self.port = dict(sorted(self.port.items()))
        self.conn_nodes = dict(sorted(self.conn_nodes.items()))
        self.conn_credential = dict(sorted(self.conn_credential.items()))
        self.coll_credential = dict(sorted(self.coll_credential.items()))
        self.pc_admin = dict(sorted(self.pc_admin.items()))
        
    def __str__(self):
        return f"""discover_nodes : {self.discover_nodes}
conn_nodes : {self.conn_nodes}
conn_credential : {self.conn_credential}
coll_credential : {self.coll_credential}
port : {self.port}
infected_pc : {self.pc_admin}"""
    
class security:
    port_max = 25000
    vulnerability_list = ["A", "B", "C", "D", "E", "F", "G"]
    
    def __init__(self, num_pc=5, seed=148, port_size=10):
        np.random.seed(seed)
        self.num_pc = num_pc
        self.pc_ip = [f"192.0.0.{i}" for i in range(num_pc)]
        self.pc_port = [np.random.randint(low=0, high=security.port_max, size=port_size) for i in range(num_pc)]
        self.pc_nodes = [np.random.choice(num_pc, np.random.randint(low=1, high=num_pc), replace=False).tolist() for i in range(num_pc)]
        self.pc_list = []
        self.attacker = attacker()
        
        for i, node in enumerate(self.pc_nodes):
            if i in node:
                node.remove(i)
                self.pc_nodes[i] = np.array(node)
            else:
                self.pc_nodes[i] = np.array(node)
        
        for i in range(num_pc):
            self.pc_list.append(PC(f"pc{i}", self.pc_ip[i], self.pc_port[i], self.pc_nodes[i]))

        PC.whole_nodes = self.pc_list
        
        for pc in self.pc_list:
            pc.env_setting()
        
    def reset(self) -> np.ndarray:
        client0 = self.pc_list[0]
        select_vulnerability = security.vulnerability_list[np.random.randint(low=0, high=len(security.vulnerability_list))]
        self.attacker.exploit_attack(client0, select_vulnerability)
        self.attacker.portscan_attack(client0)
        self.attacker.spear_phishing(client0)
        client0.open_mail()
        
        state1 = len(self.attacker.port[client0.ip])
        state2 = len(self.attacker.conn_nodes[client0.ip])
        state3 = self.attacker.key_security[client0.ip]
        state4 = self.attacker.web_cre[client0.ip]
        
        observation = np.array((state1, state2, state3, state4))
        return observation
    
    def step(self, action: int) -> Tuple[np.ndarray, int, bool, Tuple[bool, dict, bool]]:
        target_list = []
        select_vulnerability = security.vulnerability_list[np.random.randint(low=0, high=len(security.vulnerability_list))]
        
        for pc in self.attacker.discover_nodes:
            if pc not in self.attacker.pc_admin.keys() or self.attacker.pc_admin[pc] != True: 
                target_list.append(pc)
        
        if len(target_list) == 0:
            observation = np.array([0, 0, 0, 0])
            reward = 0
            done = True
            info = False, self.attacker.pc_admin ,False
            return observation, reward, done, info
        
        target = target_list[0]
        target = self.ip_to_pc(target)
        self.attacker.exploit_attack(target, select_vulnerability)
        self.attacker.portscan_attack(target)
        self.attacker.spear_phishing(target)
        target.open_mail()
        
        if target.ip in self.attacker.coll_credential.keys():
            credential = self.attacker.coll_credential[target.ip]
            self.attacker.exploit_credential(target, credential)
        
        elif self.attacker_action(action, target) and self.attacker.pc_admin[target.ip] == None: 
            credential = self.attacker.coll_credential[target.ip]
            self.attacker.exploit_credential(target, credential)

        self.attacker.list_sorting()
        if self.attacker.pc_admin[target.ip]:
            reward = 1
        else:
            reward = -1
        
        state1 = len(self.attacker.port[target.ip])
        state2 = len(self.attacker.conn_nodes[target.ip])
        state3 = self.attacker.key_security[target.ip]
        state4 = self.attacker.web_cre[target.ip]
        
        observation = np.array((state1, state2, state3, state4))
        reward = reward
        done = list(self.attacker.pc_admin.values()).count(True) == self.num_pc
        info = True, self.attacker.pc_admin, True if reward == 1 else False
        return observation, reward, done, info

    def print_env(self):
        for pc in self.pc_list:
            print(pc, end="\n\n")
            
    def attacker_action(self, action: int, target: PC) -> bool:
        if not target.infected and action == 2:
            action = 0
            
        elif not target.infected and action == 3:
            action = 1
            
        if action == 0:
            return self.attacker.try_port_access(target)
        
        elif action == 1:
            high = len(self.attacker.conn_nodes[target.ip])
            if high == 0:
                self.attacker.pc_admin[target.ip] = None
                return False
            n = np.random.randint(low=0, high=high)
            input_ip = self.attacker.conn_nodes[target.ip][n]
            return self.attacker.spoofing_login(target, input_ip)
            
        elif action == 2:
            return self.attacker.key_logging(target)
        
        elif action == 3:
            return self.attacker.access_web(target)

    def ip_to_pc(self, ip: str) -> PC:
        for pc in self.pc_list:
            if pc.ip == ip:
                return pc
    
def select_action(action: int) -> str:
    if action == 0:
        return "port_scan"
    
    elif action == 1:
        return "spoofing"
    
    elif action == 2:
        return "key_logging"
    
    elif action == 3:
        return "web_cre"
    
class success_action:
    def __init__(self):
        self.success_list = []
        self.success_action = {"port_scan": 0, "spoofing" : 0, "key_logging" : 0, "web_cre" : 0} 
    
    def put(self, action: str):
        self.success_action[action] += 1

    def reset(self):
        self.success_list.append(self.success_action)
        self.success_action = {"port_scan": 0, "spoofing" : 0, "key_logging" : 0, "web_cre" : 0} 
    
    def load_list(self) -> list:
        return self.success_list