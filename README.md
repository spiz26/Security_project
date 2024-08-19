# A Study on the Development of Adversarial Simulator for Network Vulnerability Analysis Based on Reinforcement Learning

**Journal of The Korea Institute of Information Security & Cryptology**  
**VOL.34, NO.1, Feb. 2024**  
**ISSN:** 1598-3986(Print), 2288-2715(Online)  
**DOI:** [10.13089/JKIISC.2024.34.1.21](https://doi.org/10.13089/JKIISC.2024.34.1.21)

### Authors
- **Jeongyoon Kim**  
  Seoul National University of Science & Technology (Graduate Student)
- **Jongyoul Park**  
  Seoul National University of Science & Technology (Professor)
- **Sang Ho Oh**  
  Pukyong National University (Professor)

---

### Abstract

With the development of ICT and networks, security management of IT infrastructure that has grown in size is becoming very difficult. Many companies and public institutions are having difficulty managing system and network security. In addition, as the complexity of hardware and software grows, it is becoming almost impossible for a person to manage all security. Therefore, AI is essential for network security management. However, since it is very dangerous to operate an attack model in a real network environment, cybersecurity emulation research was conducted through reinforcement learning by implementing a real-life network environment. To this end, this study applied reinforcement learning to the network environment, and as the learning progressed, the agent accurately identified the vulnerability of the network. When a network vulnerability is detected through AI, automated customized response becomes possible.

**Keywords:** Reinforcement Learning, Information Security, Network, DQN

---

# Security Simulation 환경 설정

- **운영체제**: Ubuntu 20.04
- **GPU**: NVIDIA A100 * 1
- **Driver Version**: 470.129.06
- **CUDA Version**: 11.4

- **Torch**: 1.7.1+cu110
- **Numpy**: 1.19.5
- **Matplotlib**: 3.5.2

## 프로젝트 구조

main 주피터노트북 코드가 정상작동 하기 위해서는 다음과 같은 패키지 구조가 필요합니다.

```plaintext
security_project/
    ├── RL_method/
    │   ├── RL_q_learning.py
    │   ├── RL_DQN_soft.py
    │   ├── RL_actor_critic.py
    │   └── RL_ppo.py
    ├── result_graphs/
    ├── result.ipynb
    ├── security_simulation.py
    ├── security_main_Q-learning.ipynb
    ├── security_main_DQN_soft.ipynb
    ├── security_main_AC.ipynb
    └── security_main_PPO.ipynb
```

총 10개의 파일과 하나의 디렉토리로 구성되어 있습니다.

## 파일 설명

### `security_simulation.py`
- `pc` 객체를 만들 수 있는 `pc` 클래스, 공격자 객체를 만들 수 있는 `attacker` 클래스, 환경을 만들 수 있는 `security` 클래스가 구현되어 있습니다.
- 주요 기능:
  - `pc` 객체는 네트워크 환경을 구성하는 주요 요소로서 여러 취약점을 가지고 있습니다.
  - `attacker` 객체는 `pc` 객체를 공격하는 다양한 방법을 제공합니다.
  - `security` 객체는 `pc` 객체들을 모아 네트워크 환경을 구성하고, 공격을 시뮬레이션합니다.

#### 주요 함수 및 메소드

- **`pc class`**: 각 `pc` 객체를 생성할 수 있는 클래스
- **`env_setting`**: `pc` 객체들을 생성하고 연결해주는 메소드
- **`export_credential`**: 취약점 공격 시, 연결된 `pc`들의 크리덴셜과 IP를 반환
- **`allow_access`**: 패스워드가 없는 포트로의 접근 시도를 허용
- **`allow_admin`**: 크리덴셜을 가진 접속 시도를 허용
- **`receive_mail`**: 메일 수신
- **`open_mail`**: 메일 열기 시, 바이러스 감염 여부 결정
- **`key`**: 키보드 보안 미비 시, 해킹 허용
- **`web_credential`**: 웹에 저장된 크리덴셜이 있을 시, 해킹 허용

### `security_main_AC.ipynb`
- `security` 환경을 Actor-Critic 알고리즘으로 학습시키는 메인 파일입니다.
- 에피소드 수를 지정하여 학습을 진행하며, 에피소드에 따른 누적 보상 그래프를 출력합니다.

#### 주요 함수 및 메소드

- **`main`**: Actor-Critic 알고리즘이 적용된 메인 함수

### `security_main_DQN_soft.ipynb`
- `security` 환경을 DQN 알고리즘으로 학습시키는 메인 파일입니다.
- 에피소드 수를 지정하여 학습을 진행하며, 에피소드에 따른 누적 보상 그래프를 출력합니다.

#### 주요 함수 및 메소드

- **`train`**: 학습을 진행하고 결과를 반환하는 메소드

### `security_main_Q-learning.ipynb`
- `security` 환경을 Tabular Q-learning 알고리즘으로 학습시키는 메인 파일입니다.
- 에피소드 수를 지정하여 학습을 진행하며, 에피소드에 따른 누적 보상 그래프를 출력합니다.

#### 주요 함수 및 메소드

- **`main`**: Tabular Q-learning 알고리즘이 적용된 메인 함수

### `security_main_PPO.ipynb`
- `security` 환경을 PPO 알고리즘으로 학습시키는 메인 파일입니다.
- 에피소드 수를 지정하여 학습을 진행하며, 에피소드에 따른 누적 보상 그래프를 출력합니다.

#### 주요 함수 및 메소드

- **`main`**: PPO 알고리즘이 적용된 메인 함수

### `RL_actor_critic.py`
- `security_main_AC.ipynb`에서 사용할 Actor-Critic 알고리즘이 정의된 파일입니다.
- Actor-Critic 알고리즘을 사용할 수 있도록 클래스와 메소드들이 정의되어 있습니다.

#### 주요 함수 및 메소드

- **`ActorCritic`**: Actor-Critic 클래스
- **`__init__`**: Policy net과 Value net을 정의
- **`pi`**: Policy net의 forward 메소드 (Softmax 함수 사용)
- **`v`**: Value net의 forward 메소드
- **`put_data`**: Transition 데이터를 저장하는 메소드
- **`make_batch`**: Transition 데이터를 배치 단위로 묶는 메소드
- **`train_net`**: 배치 단위의 데이터를 가지고 학습하는 메소드

### `RL_DQN_soft.py`
- `security_main_DQN_soft.ipynb`에서 사용할 DQN 알고리즘이 정의된 파일입니다.
- DQN(soft-update) 알고리즘을 사용할 수 있도록 클래스와 메소드들이 정의되어 있습니다.

#### 주요 함수 및 메소드

- **`ReplayBuffer`**: DQN에서 사용할 Replay Buffer 클래스
- **`__init__`**: Replay Buffer 초기화
- **`put`**: Transition을 Buffer에 추가
- **`sample`**: 학습에 사용할 Transition을 배치 단위로 반환

- **`Qnet`**: DQN 클래스
- **`__init__`**: 신경망 레이어 정의
- **`forward`**: Forward 메소드
- **`sample_action`**: Epsilon-Greedy 알고리즘 적용 메소드
- **`fitting_model`**: Qnet 학습 메소드
- **`train`**: DQN 학습 메인 함수

### `RL_q_learning.py`
- `security_main_Q-learning.ipynb`에서 사용할 Tabular Q-learning 알고리즘이 정의된 파일입니다.
- Tabular Q-learning 알고리즘을 사용할 수 있도록 클래스와 메소드들이 정의되어 있습니다.

#### 주요 함수 및 메소드

- **`QAgent`**: Q값이 기록된 테이블 클래스
- **`select_action`**: Action을 선택하는 메소드
- **`update_table`**: Q-learning 업데이트 메소드
- **`anneal_eps`**: Epsilon 값을 조정하는 메소드

### `RL_ppo.py`
- `security_main_PPO.ipynb`에서 사용할 PPO 알고리즘이 정의된 파일입니다.
- PPO 알고리즘을 사용할 수 있도록 클래스와 메소드들이 정의되어 있습니다.

#### 주요 함수 및 메소드

- **`PPO`**: Actor-Critic 클래스
- **`pi`**: Policy net의 forward 메소드 (Softmax 함수 사용)
- **`v`**: Value net의 forward 메소드
- **`put_data`**: Transition 데이터를 저장하는 메소드
- **`make_batch`**: Transition 데이터를 배치 단위로 묶는 메소드
- **`train_net`**: 배치 단위의 데이터를 가지고 학습하는 메소드
