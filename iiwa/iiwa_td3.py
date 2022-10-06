from machin.frame.algorithms import TD3, DDPG
from machin.utils.logging import default_logger as logger
import torch as t
import torch.nn as nn
import gym
from iiwa_env import *
from random import randrange
import random as rand

#observe_dim -> L2 norm of (target_pos - current_pos) & L1 norm of velocities
observe_dim = 10

action_dim = 7 # number of joints
action_range = t.Tensor([1.48352986, 1.48352986, 1.74532925, 1.30899694, 2.26892803, 2.35619449, 2.35619449]).to(device="cuda:0")
max_episodes = 2000
max_steps = 300
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
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        a = t.relu(self.fc3(a))
        a = t.tanh(self.fc4(a)) * 10 
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, state, action):
        state_action = t.cat([state, action], 1)
        q = t.relu(self.fc1(state_action))
        q = t.relu(self.fc2(q))
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

    td3 = TD3(
        actor,
        actor_t,
        critic,
        critic_t,
        critic2,
        critic2_t,
        t.optim.Adam,
        nn.MSELoss(reduction="sum"),
        # actor_learning_rate=0.0005,
        # critic_learning_rate=0.003,
        # discount=0.6,
        replay_device='cuda',
    )
    actor.to(d)
    actor_t.to(d)
    critic.to(d)
    critic_t.to(d)
    critic2.to(d)
    critic2_t.to(d)
    td3.enable_multiprocessing()

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

    g = False
    while episode < max_episodes:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        init_pos = []
        for i in position_limits:
            init_pos.append(rand.uniform(-i,i))
        env = Env(graphics=g, initial_positions=np.array(init_pos))
        state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim).to(d)
        tmp_observations = []

        while not terminal and step <= max_steps:
            step += 1
            if episode % 200 == 0:
               # env.render()    # pop-up a GUI of the environment in each 20th episode
               g = True
               pass
            else:
                g = False

            with t.no_grad():
                old_state = state.cpu()
                # if episode < 50:
                #     action = []
                #     for limit in action_range.cpu():
                #         action.append(rand.uniform(-limit,limit))
                #     action = t.Tensor([action])
                # else:
                action = td3.act_with_noise( {"state": old_state}, noise_param=noise_param, mode=noise_mode)
               
                state, reward, terminal, _ = env.step(action.cpu().numpy()[0])
                state = t.tensor(state, dtype=t.float32).view(1, observe_dim)
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
                state = state.to(d)

        td3.store_episode(tmp_observations)

        # update, update more if episode is longer, else less
        if episode > 10:
            for _ in range(step):
                td3.update()

        # show reward
        smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
        logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f}")

        # if smoothed_total_reward > solved_reward:
        #     reward_fulfilled += 1
        #     if reward_fulfilled >= solved_repeat:
        #         logger.info("Environment solved!")
        #         exit(0)
        # else:
        #     reward_fulfilled = 0
