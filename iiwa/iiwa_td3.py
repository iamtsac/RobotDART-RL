from machin.frame.algorithms import TD3
from machin.utils.logging import default_logger as logger
import torch
import torch.nn as nn
import gym
import json
import time
from iiwa_env import Env
from random import randrange
import random as rand
import numpy as np

#observe_dim -> L2 norm of (target_pos - current_pos) & L1 norm of velocities
observe_dim = 10

action_dim = 7 # number of joints
action_range = torch.Tensor([1.48352986, 1.48352986, 1.74532925, 1.30899694, 2.26892803, 2.35619449, 2.35619449]).to(device="cuda:0")
max_episodes = 1500
max_steps = 400
noise_param = (0, 0.2)
noise_mode = "normal"
solved_reward = -150
solved_repeat = 5

position_limits = [2.96705973, 2.0943951,  2.96705973, 2.0943951,  2.96705973, 2.0943951, 3.05432619]


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_range):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)

    def forward(self, state):
        a = torch.relu(self.fc1(state))
        a = torch.relu(self.fc2(a))
        a = torch.relu(self.fc3(a))
        a = torch.tanh(self.fc4(a)) * action_range
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        q = torch.relu(self.fc1(state_action))
        q = torch.relu(self.fc2(q))
        q = torch.relu(self.fc3(q))
        q = self.fc4(q)
        return q

if __name__ == "__main__":

    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = Actor(observe_dim, action_dim, action_range)
    actor_t = Actor(observe_dim, action_dim, action_range)
    critic = Critic(observe_dim, action_dim)
    critic_t = Critic(observe_dim, action_dim)
    critic2 = Critic(observe_dim, action_dim)
    critic2_t = Critic(observe_dim, action_dim)
    discount_factor=0.6
    td3 = TD3(
        actor,
        actor_t,
        critic,
        critic_t,
        critic2,
        critic2_t,
        torch.optim.Adam,
        nn.MSELoss(reduction="sum"),
        # actor_learning_rate=0.001,
        critic_learning_rate=0.003,
        discount=discount_factor,
        replay_device='cuda',
    )
    actor.to(d)
    actor_t.to(d)
    critic.to(d)
    critic_t.to(d)
    critic2.to(d)
    critic2_t.to(d)


    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0
    #New random initial positions
    init_pos=[]
    for i in position_limits:
        init_pos.append(rand.uniform(-i,i))
        
    try:
        data = json.loads(open("iiwa_td3.json", "r").read())
        get_run_number = int(list(data.keys())[-1].split('_')[-1]) + 1
    except:
        data = {}
        get_run_number = 1

    data[f'run_{get_run_number}'] =  {"initial_positions": init_pos, "execution_time" : None, "solved_reward": solved_reward, "episodes": {}}

    #set Environment instance
    env = Env()
    start=time.monotonic()

    prev_reward = 0
    solved_flag = False

    while episode < max_episodes:
        #Environment data
        episode += 1
        total_reward = 0
        terminal = False
        step = 0

        #Execution data for plotting
        actions = []
        rewards = []
        discounted_reward = 0
        expected_reward = []

        #New variables for reset
        g=(episode%100)==0
        tmp_observations = []

        state = torch.tensor(env.reset(init_pos, g), dtype=torch.float32).view(1, observe_dim).to(d)
        while not terminal and step <= max_steps:
            step += 1

            with torch.no_grad():
                old_state = state
                if episode < 100:
                    action = []
                    for limit in action_range.cpu():
                        action.append(rand.uniform(-limit,limit))
                    action = torch.Tensor([action])
                else:
                    action = td3.act_with_noise(
                        {"state": old_state}, noise_param=noise_param, mode=noise_mode
                        )
               
                #Data for TD3
                state, reward, terminal, _ = env.step(action.cpu().numpy()[0])
                state = torch.tensor(state, dtype=torch.float32).view(1, observe_dim).to(d)
                total_reward += reward
                tmp_observations.append(
                    {
                        "state": {"state": old_state},
                        "action": {"action": action},
                        "next_state": {"state": state},
                        "reward": reward,
                        "terminal": terminal or step == max_steps,
                    }
                )
                #Data for plotting
                discounted_reward += (discount_factor ** step) * reward
                actions.append(action.cpu().numpy()[0].tolist())
                rewards.append(reward)


        expected_reward.append(discounted_reward/max_steps)
        td3.store_episode(tmp_observations)

        # update, update more if episode is longer, else less
        if episode > 100:
            for _ in range(step):
                td3.update()

        # show reward
        smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
        logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f}")

        if (np.floor(smoothed_total_reward/10) == np.floor(prev_reward/10)) and not solved_flag:
            reward_fulfilled += 1
            data[f'run_{get_run_number}']['episodes'][episode] = {"total_reward": None,"actions": None, "rewards": None}
            data[f'run_{get_run_number}']['episodes'][episode]["total_reward"] = total_reward
            data[f'run_{get_run_number}']['episodes'][episode]["actions"] = actions
            data[f'run_{get_run_number}']['episodes'][episode]["rewards"] = rewards
            if reward_fulfilled >= solved_repeat:
                logger.info("Environment solved!")
                data[f'run_{get_run_number}']['execution_time'] = time.monotonic() - start
                data[f'run_{get_run_number}']['solved_reward'] = np.floor(smoothed_total_reward/10) * 10 
                solved_flag = True
        else:
            reward_fulfilled = 0
            prev_reward = smoothed_total_reward

    s = json.dumps(data)
    open("iiwa_td3.json","w").write(s)
