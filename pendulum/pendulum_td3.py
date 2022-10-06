import time
import json
import matplotlib.pyplot as plt
import torch 
from torch.distributions import Normal
import torch.nn as nn
from pendulum_env import * 
from machin.frame.algorithms import TD3
from machin.utils.logging import default_logger as logger

# configurations
env = Env()
observe_dim = 3
action_dim = 1
max_episodes = 5000
max_steps = 200
solved_reward = -290
solved_repeat = 20
noise_param = (0, .2)
noise_mode = "normal"
action_range = 2.5
discount_factor = .99


# model definition
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_range):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)
        self.action_range = action_range

    def forward(self, state):
        a = torch.relu(self.fc1(state))
        a = torch.relu(self.fc2(a))
        a = torch.tanh(self.fc3(a))  * self.action_range
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        q = torch.relu(self.fc1(state_action))
        q = torch.relu(self.fc2(q))
        q = self.fc3(q)
        return q




if __name__ == "__main__":

    actor = Actor(observe_dim, action_dim, action_range)
    actor_t = Actor(observe_dim, action_dim, action_range)
    critic = Critic(observe_dim, action_dim)
    critic_t = Critic(observe_dim, action_dim)
    critic2 = Critic(observe_dim, action_dim)
    critic2_t = Critic(observe_dim, action_dim)

    td3 = TD3(
        actor,
        actor_t,
        critic,
        critic_t,
        critic2,
        critic2_t,
        torch.optim.Adam,
        nn.MSELoss(reduction="sum"),
        actor_learning_rate = 0.01,
    )


    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0
    render = False

    expected_reward = []
    try:
        data = json.loads(open("pendulum_td3.json", "r").read())
        get_run_number = int(list(data.keys())[-1].split('_')[-1]) + 1
    except:
        data = {}
        get_run_number = 1

    data[f'run_{get_run_number}'] =  {"execution_time" : None, "solved_reward": solved_reward, "episodes": {}}
    start = time.monotonic()

    while episode < max_episodes:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        render = not (episode % 200)
        state = torch.tensor(env.reset(initial_positions=[np.pi], render=render), dtype=torch.float32).view(1, observe_dim)
        tmp_observations = []
        discounted_reward = 0
        actions = []
        rewards = []
        data[f'run_{get_run_number}']['episodes'][episode] = {"total_reward": None,"actions": None, "rewards": None}

        while step <= max_steps:
            step += 1
            with torch.no_grad():
                old_state = state
                # Run 100 random action to explore the enviroment.
                if episode < 100: 
                    action = torch.tensor(np.array([torch.FloatTensor(1,).uniform_(-2.5, 2.5).numpy()], dtype=np.float32))
                else:
                    action = td3.act_with_noise(
                        {"state": old_state}, noise_param=noise_param, mode=noise_mode
                    )
               
                state, reward, terminal, _ = env.step(action.numpy())
                state = torch.tensor(state, dtype=torch.float32).view(1, observe_dim)
                total_reward += reward
                discounted_reward += (discount_factor ** step) * reward
                tmp_observations.append(
                    {
                        "state": {"state": old_state},
                        "action": {"action": action},
                        "next_state": {"state": state},
                        "reward": reward,
                        "terminal": terminal or step == max_steps,
                    }
                )

                actions.append(action.cpu().numpy().item())
                rewards.append(reward)
        data[f'run_{get_run_number}']['episodes'][episode]["total_reward"] = total_reward
        data[f'run_{get_run_number}']['episodes'][episode]["actions"] = actions
        data[f'run_{get_run_number}']['episodes'][episode]["rewards"] = rewards
        expected_reward.append(discounted_reward/max_steps)
        td3.store_episode(tmp_observations)

        # update, update more if episode is longer, else less
        if episode > 100:
            for _ in range(step):
                td3.update()

        # show reward
        smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
        logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f}")

        if smoothed_total_reward > solved_reward:
            reward_fulfilled += 1
            if reward_fulfilled >= solved_repeat:
                logger.info("Environment solved!")
                data[f'run_{get_run_number}']['execution_time'] = time.monotonic() - start
                s = json.dumps(data)
                open("pendulum_td3.json","w").write(s)
                exit(0)
        else:
            reward_fulfilled = 0