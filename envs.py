from typing import Optional

import gym
import numpy as np
from ray.tune import register_env

OBS = ['HR', 'BP', 'T', 'D']
OBS_MIN = np.array([0, 0, 0, 0])
OBS_MAX = np.array([200, 200, 40, 10])
MIN_ALLOWED = np.array([60, 80, 36, -0.1])
MAX_ALLOWED = np.array([110, 130, 38, 1.1])
DONE_MIN = np.array([55, 75, 35.3, -0.1])
DONE_MAX = np.array([140, 140, 38.7, 1.1])
ACTION_MAX = np.array([0.5])
ACTION_MIN = np.array([0])
MAX_TIME_STEPS = 500


class DosingEnv(gym.Env):

    def __init__(self):
        self.action_space = gym.spaces.Box(low=ACTION_MIN, high=ACTION_MAX)
        self.observation_space = gym.spaces.Box(low=OBS_MIN, high=OBS_MAX)

        self.DELTA_T = 1
        self.heart_rate = 70
        self.blood_pressure = 120
        self.temperature = 37
        self.dosage = 0
        self.BP_normal = 120
        self.HR_normal = 70
        self.T_normal = 37
        self.t = 0

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        self.heart_rate = 70
        self.blood_pressure = 120
        self.temperature = 37
        self.dosage = 0.1
        self.t = 0
        return np.array([self.heart_rate, self.blood_pressure, self.temperature, self.dosage])

    def step(self, action):
        action = action[0]
        prev_heart_rate = self.heart_rate
        prev_blood_pressure = self.blood_pressure
        prev_temperature = self.temperature
        prev_dosage = self.dosage

        self.dosage = action + (0.02 * (1 - prev_dosage)) * self.DELTA_T
        if self.dosage < 0:
            self.dosage = 0
        self.heart_rate = prev_heart_rate + (0.3 * (prev_blood_pressure - self.BP_normal) - 0.05 * (
                prev_heart_rate - self.HR_normal) - 0.2 * prev_dosage) * self.DELTA_T
        self.blood_pressure = prev_blood_pressure + (0.15 * (prev_heart_rate - self.HR_normal) - 0.05 * (
                prev_blood_pressure - self.BP_normal) - 0.3 * prev_dosage) * self.DELTA_T
        self.temperature = prev_temperature + (0.05 * (prev_blood_pressure - self.BP_normal) - 0.006 * (
                prev_temperature - self.T_normal) - 0.1 * prev_dosage) * self.DELTA_T

        obs = np.array([self.heart_rate, self.blood_pressure, self.temperature, self.dosage])

        reward = np.sum((MIN_ALLOWED <= obs) & (MIN_ALLOWED <= obs)) * 0.25 - 0.25

        self.t = self.t + 1
        print(f'Action: {action}')
        print(obs)
        print((MIN_ALLOWED <= obs) & (MIN_ALLOWED <= obs))
        print((MIN_ALLOWED > obs) | (MAX_ALLOWED < obs))
        print(self.t)
        print(reward)
        done = np.any((MIN_ALLOWED > obs) | (MAX_ALLOWED < obs)) and (self.t < MAX_TIME_STEPS)

        return obs, reward, done, {}

    def render(self, mode="human"):
        print(f'Current Heart Rate: {self.heart_rate}')
        print(f'Current Blood Pressure: {self.blood_pressure}')
        print(f'Current Temperature: {self.temperature}')
        print(f'Current Dosage: {self.dosage}')


def env_creator(env_config):
    return DosingEnv()


def register_env_dosage(env_name):
    register_env(env_name, env_creator)


env = DosingEnv()
obs = env.reset()

done = False

while not done:
    action = env.action_space.sample()
    print(action)
    obs, reward, done, _ = env.step([0])
    env.render()
