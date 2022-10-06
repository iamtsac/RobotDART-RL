from machin.frame.algorithms import TD3, DDPG
from machin.utils.logging import default_logger as logger
import torch as t
import torch.nn as nn
import gym
from pendulum_env import * 

# configurations
#env = gym.make("Pendulum-v1")
observe_dim = 3
action_dim = 1
action_range = 2
max_episodes = 2000
max_steps = 120
noise_param = (0, 0.2)
noise_mode = "normal"
solved_reward = -150
solved_repeat = 5


# model definition
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_range):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)
        self.action_range = action_range

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        a = t.tanh(self.fc3(a)) * 5 * self.action_range
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, state, action):
        state_action = t.cat([state, action], 1)
        q = t.relu(self.fc1(state_action))
        q = t.relu(self.fc2(q))
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
        t.optim.Adam,
        nn.MSELoss(reduction="sum"),
        actor_learning_rate = 0.01,
    )


    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

    g = False
    while episode < max_episodes:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        env = Env(graphics=g, initial_positions=[np.pi])
        state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)
        tmp_observations = []

        while not terminal and step <= max_steps:
            step += 1
            if episode % 100 == 0:
               # env.render()    # pop-up a GUI of the environment in each 20th episode
               g = True
               pass
            else:
                g = False

            with t.no_grad():
                old_state = state
                #agent model inference
                if episode < 100: 
                    action = t.tensor(np.array([t.FloatTensor(1,).uniform_(-5, 5).numpy()], dtype=np.float32))
                else:
                    action = td3.act_with_noise(
                        {"state": old_state}, noise_param=noise_param, mode=noise_mode
                    )
               
                state, reward, terminal, _ = env.step(action.numpy())
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
                # if tmp_observations[-1]['terminal']:
                #     print(tmp_observations[-1]['state'])

        td3.store_episode(tmp_observations)

        # update, update more if episode is longer, else less
        if episode > 120:
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
