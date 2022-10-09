import time
from machin.frame.algorithms import PPO
from machin.utils.logging import default_logger as logger
import torch 
from torch.distributions import Normal
import json
from iiwa_env import * 
import numpy as np
import matplotlib.pyplot as plt
import random as rand

# configurations
observe_dim = 10
action_dim = 7
max_episodes = 2500
max_steps = 400
noise_param = (0, 0.2)
noise_mode = "normal"
solved_reward = -100
solved_repeat = 10
position_limits = [2.96705973, 2.0943951,  2.96705973, 2.0943951,  2.96705973, 2.0943951, 3.05432619]

# model definition
class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.h1 = torch.nn.Linear(state_dim, 128)
        self.h2 = torch.nn.Linear(128, 64)
        self.h3 = torch.nn.Linear(64, 32)
        self.out = torch.nn.Linear(32, action_dim)
        self.out_sigma = torch.nn.Linear(32, action_dim)

    def forward(self, state, action=None):
        a = torch.relu(self.h1(state))
        a = torch.relu(self.h2(a))
        a = torch.relu(self.h3(a))
        mu = self.out(a)
        log_sigma = torch.nn.functional.softplus(self.out_sigma(a))
        dist = Normal(mu, torch.exp(log_sigma))
        actions = action if action is not None else dist.sample()
        action_entropy = dist.entropy()
        action_log_prob = dist.log_prob(actions)
        return actions, action_log_prob, action_entropy

class Critic(torch.nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        self.h1 = torch.nn.Linear(state_dim, 128)
        self.h2 = torch.nn.Linear(128, 64)
        self.h3 = torch.nn.Linear(64, 32)
        self.h4 = torch.nn.Linear(32, 1)

    def forward(self, state):
        v = torch.relu(self.h1(state))
        v = torch.relu(self.h2(v))
        v = torch.relu(self.h3(v))
        v = self.h4(v)
        return v

if __name__ == "__main__":

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = Actor(observe_dim, action_dim)
    critic = Critic(observe_dim)
    actor.to(dev)
    critic.to(dev)
    discount_factor = 0.4
    ppo = PPO(
        actor, 
        critic,
        torch.optim.Adam, 
        torch.nn.MSELoss(reduction="sum"),
        discount = discount_factor,
        replay_device=dev,
        )

 

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0
    init_pos=[]
    for i in position_limits:
        init_pos.append(rand.uniform(-i,i))
        
    try:
        data = json.loads(open("iiwa_ppo.json", "r").read())
        get_run_number = int(list(data.keys())[-1].split('_')[-1]) + 1
    except:
        data = {}
        get_run_number = 1

    data[f'run_{get_run_number}'] =  {"initial_positions": init_pos, "execution_time" : None, "solved_reward": solved_reward, "episodes": {}}



    #Create environment instance
    env=Env()
    start = time.monotonic()
    init_pos = []

    for i in position_limits:
        init_pos.append(np.random.uniform(-i,i))

    prev_reward = 0
    last_actions = []

    while episode < max_episodes:
        #Environment variables
        episode += 1
        total_reward = 0
        terminal = False
        step = 0

        #Execution data for plotting
        actions = []
        rewards = []
        data[f'run_{get_run_number}']['episodes'][episode] = {"total_reward": None,"actions": None, "rewards": None}

        #New variables for reset 
        render = not (episode % 100)
        tmp_observations = []

        #Reset environment
        state = torch.tensor(env.reset(initial_positions=init_pos, render=render), dtype=torch.float32).view(1, observe_dim).to(dev)

        while not terminal and step <= max_steps:
            step += 1

            with torch.no_grad():
                #Data for PPO
                old_state = state
                action = ppo.act({"state": old_state})[0].view(1,7)
                state, reward, terminal, _ = env.step(action.cpu().numpy()[0])
                state = torch.tensor(state, dtype=torch.float32).view(1, observe_dim)
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
                actions.append(action.cpu().numpy()[0].tolist())
                rewards.append(reward)

        data[f'run_{get_run_number}']['episodes'][episode]["total_reward"] = total_reward
        data[f'run_{get_run_number}']['episodes'][episode]["rewards"] = rewards

        ppo.store_episode(tmp_observations)
        ppo.update(concatenate_samples=True)
    

        # show reward
        smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
        logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f}")

        #Check if environment is solved
        if (np.floor(smoothed_total_reward/10) == np.floor(prev_reward/10)):
            reward_fulfilled += 1
            last_actions.append(actions)
            if reward_fulfilled >= solved_repeat:
                logger.info("Environment solved!")
                # Store last actions
                for i in range(solved_repeat):
                    data[f'run_{get_run_number}']['episodes'][episode - solved_repeat + i + 1]["actions"] = last_actions[i]
                data[f'run_{get_run_number}']['execution_time'] = time.monotonic() - start
                data[f'run_{get_run_number}']['solved_reward'] = np.floor(smoothed_total_reward/10) * 10 
                s = json.dumps(data)
                open("iiwa_ppo.json","w").write(s)
                exit(0)
        else:
            reward_fulfilled = 0
            prev_reward = smoothed_total_reward
            last_actions = []

