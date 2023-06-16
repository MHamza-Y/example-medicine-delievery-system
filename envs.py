from typing import Optional

import gym
import numpy as np
from ray.tune import register_env

# OBS = ['HR', 'BP', 'T', 'D']
# OBS_MIN = np.array([0, 0, 0, 0])
# OBS_MAX = np.array([200, 200, 40, 10])
# MIN_ALLOWED = np.array([60, 80, 36, -0.1])
# MAX_ALLOWED = np.array([110, 130, 38, 10])
# DONE_MIN = np.array([55, 75, 35.3, -0.1])
# DONE_MAX = np.array([140, 140, 38.7, 10])
ACTION_MAX = np.array([0.5])
ACTION_MIN = np.array([0])
MAX_TIME_STEPS = 500
#
#
# class DosingEnv(gym.Env):
#
#     def __init__(self):
#         self.action_space = gym.spaces.Box(low=ACTION_MIN, high=ACTION_MAX)
#         self.observation_space = gym.spaces.Box(low=OBS_MIN, high=OBS_MAX)
#
#         self.DELTA_T = 5
#         self.heart_rate = 70
#         self.blood_pressure = 120
#         self.temperature = 37
#         self.dosage = 0
#         self.BP_normal = 120
#         self.HR_normal = 70
#         self.T_normal = 37
#         self.t = 0
#
#     def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
#         self.heart_rate = 70
#         self.blood_pressure = 120
#         self.temperature = 37
#         self.dosage = 0.1
#         self.t = 0
#         return np.array([self.heart_rate, self.blood_pressure, self.temperature, self.dosage])
#
#     def step(self, action):
#         action = action[0]
#         prev_heart_rate = self.heart_rate
#         prev_blood_pressure = self.blood_pressure
#         prev_temperature = self.temperature
#         prev_dosage = self.dosage
#
#         self.dosage = action + (0.02 * (1 - prev_dosage)) * self.DELTA_T
#         if self.dosage < 0:
#             self.dosage = 0
#
#         self.heart_rate = prev_heart_rate + (0.3 * (prev_blood_pressure - self.BP_normal) - 0.05 * (
#                 prev_heart_rate - self.HR_normal) - 0.2 * prev_dosage) * self.DELTA_T
#         self.blood_pressure = prev_blood_pressure + (0.15 * (prev_heart_rate - self.HR_normal) - 0.05 * (
#                 prev_blood_pressure - self.BP_normal) - 0.3 * prev_dosage) * self.DELTA_T
#         self.temperature = prev_temperature + (0.05 * (prev_blood_pressure - self.BP_normal) - 0.006 * (
#                 prev_temperature - self.T_normal) - 0.1 * prev_dosage) * self.DELTA_T
#
#         obs = np.array([self.heart_rate, self.blood_pressure, self.temperature, self.dosage])
#
#         reward = np.sum((MIN_ALLOWED <= obs) & (MIN_ALLOWED <= obs)) * 0.25 - 0.25
#
#         self.t = self.t + 1
#         print(f'Action: {action}')
#         print(obs)
#         print((MIN_ALLOWED <= obs) & (MIN_ALLOWED <= obs))
#         print((MIN_ALLOWED > obs) | (MAX_ALLOWED < obs))
#         print(self.t)
#         print(reward)
#         done = np.any((DONE_MIN > obs) | (DONE_MAX < obs)) | (self.t >= MAX_TIME_STEPS)
#
#         return obs, reward, done, {}
#
#     def render(self, mode="human"):
#         print(f'Current Heart Rate: {self.heart_rate}')
#         print(f'Current Blood Pressure: {self.blood_pressure}')
#         print(f'Current Temperature: {self.temperature}')
#         print(f'Current Dosage: {self.dosage}')


OBS2 = ['HR', 'SBP', 'DBP', 'T', 'Conc']
OBS_MIN2 = np.array([0, 0, 0, 0, 0])
OBS_MAX2 = np.array([200, 200, 200, 42, 10])
MIN_ALLOWED2 = np.array([60, 90, 60, 36.5])  # for rewards
MAX_ALLOWED2 = np.array([100, 120, 80, 37.5])
DONE_MIN2 = np.array([40, 90, 60, 35, 0])
DONE_MAX2 = np.array([200, 200, 120, 41, 10])

MEAN_POINT = np.array([80, 105, 70, 37])


class DosingEnvV2(gym.Env):

    def __init__(self):
        self.action_space = gym.spaces.Box(low=ACTION_MIN, high=ACTION_MAX)
        self.observation_space = gym.spaces.Box(low=OBS_MIN2, high=OBS_MAX2)
        self.heart_rate = 70
        self.sbp = 120
        self.dbp = 80
        self.temperature = 37
        self.conc = 0

        self.dt = .1

        self.k1 = 0.05
        self.k2 = 0.2
        self.k3 = 0.1
        self.k4 = 0.08
        self.k5 = 0.03

        self.t = 0

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        self.heart_rate = 75
        self.sbp = 110
        self.dbp = 85
        self.temperature = 36.5
        self.conc = 0

        self.t = 0

        return np.array([self.heart_rate, self.sbp, self.dbp, self.temperature, self.conc])

    def step(self, action):
        action = action[0]
        prev_hr = self.heart_rate
        prev_sbp = self.sbp
        prev_dbp = self.dbp
        prev_temperature = self.temperature
        prev_conc = self.conc

        self.conc = prev_conc + self.k1 * action * self.dt
        self.heart_rate = prev_hr + (self.k2 * prev_conc - (prev_hr - 70)) * self.dt
        self.sbp = prev_sbp + (self.k3 * prev_conc - (prev_sbp - 120)) * self.dt
        self.dbp = prev_dbp + (self.k4 * prev_conc - (prev_dbp - 80)) * self.dt
        self.temperature = prev_temperature + (self.k5 * prev_conc - (prev_temperature - 37)) * self.dt

        obs = np.array([self.heart_rate, self.sbp, self.dbp, self.temperature, self.conc])
        #reward = np.sum((MIN_ALLOWED2 <= obs) & (MAX_ALLOWED2 >= obs)) * 0.25 - 0.25

        deviation_from_mean = np.abs((obs[0:4] - MEAN_POINT))/(MAX_ALLOWED2 - MIN_ALLOWED2)
        print(deviation_from_mean)
        reward = np.sum(deviation_from_mean)
        done = np.any((DONE_MIN2 > obs) | (DONE_MAX2 < obs)) | (self.t >= MAX_TIME_STEPS)

        self.t += 1
        print(reward)
        print(self.t)
        return obs, reward, done, {}

    def render(self, mode="human"):
        print(f'Current Heart Rate: {self.heart_rate}')
        print(f'Current Sys Blood Pressure: {self.sbp}')
        print(f'Current Dia Blood Pressure: {self.dbp}')
        print(f'Current Temperature: {self.temperature}')
        print(f'Current Dose Concentration: {self.conc}')


def env_creator(env_config):
    return DosingEnvV2()


def register_env_dosage(env_name):
    register_env(env_name, env_creator)


env = DosingEnvV2()
obs = env.reset()

done = False
rewards = 0
while not done:
    action = env.action_space.sample()
    print(action)
    obs, reward, done, _ = env.step([0])
    rewards += reward
    env.render()

print(rewards)
