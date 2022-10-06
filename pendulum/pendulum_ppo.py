import time
import json
import matplotlib.pyplot as plt
import torch 
from torch.distributions import Normal, Categorical
from pendulum_env import * 
from machin.frame.algorithms import PPO
from machin.utils.logging import default_logger as logger

# configurations
env = Env()
observe_dim = 3
action_dim = 1
max_episodes = 5000
max_steps = 200
solved_reward = -650
solved_repeat = 20
discount_factor = .8


# model definition
class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.h1 = torch.nn.Linear(state_dim, 32)
        self.h2 = torch.nn.Linear(32, 16)
        self.out = torch.nn.Linear(16, action_dim)
        self.out_sigma = torch.nn.Linear(16, action_dim)

    def forward(self, state, action=None):
        a = torch.relu(self.h1(state))
        a = torch.relu(self.h2(a))
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

        self.h1 = torch.nn.Linear(state_dim, 64)
        self.h2 = torch.nn.Linear(64, 32)
        self.h3 = torch.nn.Linear(32, 1)

    def forward(self, state):
        v = torch.relu(self.h1(state))
        v = torch.relu(self.h2(v))
        v = self.h3(v)
        return v

if __name__ == "__main__":

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = Actor(observe_dim, action_dim)
    critic = Critic(observe_dim)
    actor.to(dev)
    critic.to(dev)

    ppo = PPO(
        actor, 
        critic,
        torch.optim.Adam, 
        torch.nn.MSELoss(reduction="sum"),
        critic_learning_rate = 0.005,
        discount = discount_factor
        )

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0
    render = False

    expected_reward = []
    try:
        data = json.loads(open("pendulum_ppo.json", "r").read())
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
                action = ppo.act({"state": old_state})[0]
                state, reward, terminal, _ = env.step(action.cpu().numpy())
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
        ppo.store_episode(tmp_observations)
        ppo.update()

        # show reward
        smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
        logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f}")

        if smoothed_total_reward > solved_reward:
            reward_fulfilled += 1
            if reward_fulfilled >= solved_repeat:
                logger.info("Environment solved!")
                data[f'run_{get_run_number}']['execution_time'] = time.monotonic() - start
                s = json.dumps(data)
                open("pendulum_ppo.json","w").write(s)
                exit(0)
        else:
            reward_fulfilled = 0