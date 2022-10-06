import sys
import json
import numpy as np
import matplotlib.pyplot as plt

PENDULUM_PPO_DISCOUNT = 0.99
PENDULUM_TD3_DISCOUNT = 0.99
IIWA_PPO_DISCOUNT = 0.99
IIWA_TD3_DISCOUNT = 0.99

expected_reward = []
data = json.loads(open('pendulum/pendulum_ppo.json', "r").read())
max_len = 0
for run in range(1,6):
    tmp_expected_reward = []
    for k, v in data[f'run_{run}']['episodes'].items():
        discounted = 0
        for t, reward in enumerate(v['rewards']):
            discounted += (PENDULUM_PPO_DISCOUNT ** t) * reward
        tmp_expected_reward.append(discounted/len(v['rewards']))
    max_len = max(max_len, len(tmp_expected_reward))
    expected_reward.append(np.asarray(tmp_expected_reward[:498], dtype=np.float32))
     

expected_reward = np.asarray(expected_reward, dtype=object)
plt.plot(np.median(expected_reward, axis=0))
plt.show()

# plt.plot(expected_reward[0])
# plt.plot(expected_reward[1])
# plt.plot(expected_reward[2])
# plt.show()

execution_time1 = []
execution_time2 = []
data_ppo = data
data_td3 = json.loads(open('pendulum/pendulum_td3.json', "r").read())
for run in range(1,6):
    execution_time1.append(data_ppo[f'run_{run}']['execution_time'])
    execution_time2.append(data_td3[f'run_{run}']['execution_time'])

plt.boxplot([execution_time1, execution_time2])

expected_reward = []
data = json.loads(open('pendulum/pendulum_ppo.json', "r").read())
max_len = 0
for run in range(1,2):
    tmp_expected_reward = []
    for k, v in data[f'run_{run}']['episodes'].items():
        discounted = 0
        for t, reward in enumerate(v['rewards']):
            discounted += (0.99 ** t) * reward
        tmp_expected_reward.append(discounted/len(v['rewards']))
    max_len = max(max_len, len(tmp_expected_reward))
    expected_reward.append(np.asarray(tmp_expected_reward, dtype=np.float32))

# for i, j in enumerate(expected_reward):
#     expected_reward[i] = np.pad(j, (0,max_len - len(j)), 'constant', constant_values=0)
     

expected_reward = np.asarray(expected_reward, dtype=object)
plt.plot(np.median(expected_reward, axis=0))
# plt.plot(np.quantile(expected_reward,0.25, axis=0))
# plt.plot(np.quantile(expected_reward,0.75, axis=0))
# plt.plot(np.quantile(expected_reward,1, axis=0))
# plt.plot(np.quantile(expected_reward,0, axis=0))
plt.fill_between(np.arange(813), np.quantile(expected_reward,0, axis=0).astype(np.float32), np.quantile(expected_reward,1, axis=0).astype(np.float32), facecolor='red', alpha=0.5)

plt.show()