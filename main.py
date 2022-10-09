from statistics import quantiles
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

PENDULUM_PPO_DISCOUNT = 0.8
PENDULUM_TD3_DISCOUNT = 0.99
IIWA_PPO_DISCOUNT = 0.4
IIWA_TD3_DISCOUNT = 0.7


"""
    Pendulum Plots
"""
#Expected reward plotting for Pendulum PPO 
expected_reward = []
data_pendulum_ppo = json.loads(open('pendulum/pendulum_ppo.json', "r").read())
max_len = 0
for run in range(1,6):
    tmp_expected_reward = []
    for k, v in data_pendulum_ppo[f'run_{run}']['episodes'].items():
        discounted = 0
        for t, reward in enumerate(v['rewards']):
            discounted += (PENDULUM_PPO_DISCOUNT ** t) * reward
        tmp_expected_reward.append(discounted/len(v['rewards']))
    max_len = max(max_len, len(tmp_expected_reward))
    expected_reward.append(np.asarray(tmp_expected_reward, dtype=np.float32))

for i, j in enumerate(expected_reward):
     expected_reward[i] = np.pad(j, (0,max_len - len(j)), 'reflect')
     
expected_reward = np.asarray(expected_reward, dtype=object)
plt.title("Pendulum PPO - Expected Reward per episode")
plt.plot(np.median(expected_reward, axis=0))
plt.fill_between(np.arange(max_len), np.quantile(expected_reward,0.25, axis=0).astype(np.float32), np.quantile(expected_reward,0.75, axis=0).astype(np.float32), facecolor='red', alpha=0.5)

plt.show()

#Expected reward plotting for Pendulum TD3 
expected_reward = []
data_pendulum_td3= json.loads(open('pendulum/pendulum_td3.json', "r").read())
max_len = 0
for run in range(1,6):
    tmp_expected_reward = []
    for k, v in data_pendulum_td3[f'run_{run}']['episodes'].items():
        discounted = 0
        for t, reward in enumerate(v['rewards']):
            discounted += (PENDULUM_TD3_DISCOUNT ** t) * reward
        tmp_expected_reward.append(discounted/len(v['rewards']))
    max_len = max(max_len, len(tmp_expected_reward))
    expected_reward.append(np.asarray(tmp_expected_reward, dtype=np.float32))

for i, j in enumerate(expected_reward):
     expected_reward[i] = np.pad(j, (0,max_len - len(j)), 'reflect')
     
expected_reward = np.asarray(expected_reward, dtype=object)
plt.title("Pendulum TD3 - Expected Reward per episode")
plt.plot(np.median(expected_reward, axis=0))
plt.fill_between(np.arange(max_len), np.quantile(expected_reward,0.25, axis=0).astype(np.float32), np.quantile(expected_reward,0.75, axis=0).astype(np.float32), facecolor='red', alpha=0.5)

plt.show()



#Execution time comparison for pendulum

execution_time_pend_ppo = []
execution_time_pend_td3 = []
for run in range(1,6):
    execution_time_pend_ppo.append(data_pendulum_ppo[f'run_{run}']['execution_time'])
    execution_time_pend_td3.append(data_pendulum_td3[f'run_{run}']['execution_time'])
plt.title("Execution time for Pendulum per algorithm")
plt.boxplot(labels=["PPO", "TD3"],x=[execution_time_pend_ppo, execution_time_pend_td3])
plt.ylabel("Seconds")
plt.show()



"""
    Iiwa Plots
"""
#Expected reward plotting for Iiwa PPO 
expected_reward = []
data_iiwa_ppo = json.loads(open('iiwa_ppo.json', "r").read())
max_len = 0
for run in range(1,6):
    tmp_expected_reward = []
    for k, v in data_iiwa_ppo[f'run_{run}']['episodes'].items():
        discounted = 0
        for t, reward in enumerate(v['rewards']):
            discounted += (IIWA_PPO_DISCOUNT ** t) * reward
        tmp_expected_reward.append(discounted/len(v['rewards']))
    max_len = max(max_len, len(tmp_expected_reward))
    expected_reward.append(np.asarray(tmp_expected_reward, dtype=np.float32))

for i, j in enumerate(expected_reward):
     expected_reward[i] = np.pad(j, (0,max_len - len(j)), 'reflect')
     
expected_reward = np.asarray(expected_reward, dtype=object)
plt.title("Iiwa PPO - Expected Reward per episode")
plt.plot(np.median(expected_reward, axis=0))
plt.fill_between(np.arange(max_len), np.quantile(expected_reward,0.25, axis=0).astype(np.float32), np.quantile(expected_reward,0.75, axis=0).astype(np.float32), facecolor='red', alpha=0.5)

plt.show()

#Expected reward plotting for Iiwa TD3 
expected_reward = []
data_iiwa_td3 = json.loads(open('iiwa/iiwa_td3.json', "r").read())
max_len = 0
for run in range(1,6):
    tmp_expected_reward = []
    for k, v in data_iiwa_td3[f'run_{run}']['episodes'].items():
        discounted = 0
        for t, reward in enumerate(v['rewards']):
            discounted += (IIWA_TD3_DISCOUNT ** t) * reward
        tmp_expected_reward.append(discounted/len(v['rewards']))
    max_len = max(max_len, len(tmp_expected_reward))
    expected_reward.append(np.asarray(tmp_expected_reward, dtype=np.float32))

for i, j in enumerate(expected_reward):
     expected_reward[i] = np.pad(j, (0,max_len - len(j)), 'reflect')
     
expected_reward = np.asarray(expected_reward, dtype=object)
plt.title("Iiwa TD3 - Expected Reward per episode")
plt.plot(np.median(expected_reward, axis=0))
plt.fill_between(np.arange(max_len), np.quantile(expected_reward,0.25, axis=0).astype(np.float32), np.quantile(expected_reward,0.75, axis=0).astype(np.float32), facecolor='red', alpha=0.5)

plt.show()



#Execution time comparison for Iiwa

execution_time_iiwa_ppo = []
execution_time_iiwa_td3 = []
for run in range(1,6):
    execution_time_iiwa_ppo.append(data_iiwa_ppo[f'run_{run}']['execution_time'])
    execution_time_iiwa_td3.append(data_iiwa_td3[f'run_{run}']['execution_time'])
plt.title("Execution time for Iiwa per algorithm")
plt.boxplot(labels=["PPO", "TD3"],x=[execution_time_iiwa_ppo, execution_time_iiwa_td3])
plt.ylabel("Seconds")
plt.show()