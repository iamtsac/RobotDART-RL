from machin.frame.algorithms import PPO
from machin.utils.logging import default_logger as logger
import torch 
from torch.distributions import Normal, Categorical
import gym
from pendulum_env import * 

# configurations
#env = gym.make("Pendulum-v1")
observe_dim = 3
action_dim = 1
max_episodes = 5000
max_steps = 80
noise_param = (0, 0.2)
noise_mode = "normal"
solved_reward = 150
solved_repeat = 5


# model definition
class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.h1 = torch.nn.Linear(state_dim, 64)
        self.h2 = torch.nn.Linear(64, 32)
        self.out = torch.nn.Linear(32, action_dim)
        self.out_sigma = torch.nn.Linear(32, action_dim)

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

    actor = Actor(observe_dim, action_dim)
    critic = Critic(observe_dim)

    replay_buffer = machin.frame.buffers.buffer.Buffer(100)
    ppo = PPO(
        actor, 
        critic,
        torch.optim.Adam, 
        torch.nn.MSELoss(reduction="sum"),
        critic_learning_rate = 0.01,
        replay_size = 200,
        discount = 0.7
        )
    ppo.enable_multiprocessing()

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

    g = False
    while episode < max_episodes:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        env = Env(graphics=g, initial_positions=[np.pi])
        state = torch.tensor(env.reset(), dtype=torch.float32).view(1, observe_dim)
        tmp_observations = []

        while not terminal and step <= max_steps:
            step += 1
            if episode % 100 == 0:
               # env.render()    # pop-up a GUI of the environment in each 20th episode
               g = True
               pass
            else:
                g = False

            with torch.no_grad():
                old_state = state
                action = ppo.act({"state": old_state})[0]
                state, reward, terminal, _ = env.step(action.numpy())
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
                # if tmp_observations[-1]['terminal']:
                #     print(tmp_observations[-1]['action'])

        ppo.store_episode(tmp_observations)
        ppo.update()

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
