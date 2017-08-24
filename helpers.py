import numpy as np
import deepmind_lab
from skimage.color import rgb2gray
import time
import ujson as json


def _action(*entries):
  return np.array(entries, dtype=np.intc)

ACTION_LIST = [
  _action(-20, 0, 0, 0, 0, 0, 0),
  _action(20, 0, 0, 0, 0, 0, 0),
  _action(0, 0, 0, 1, 0, 0, 0),
  _action(0, 0, 0, -1, 0, 0, 0),
]


class JSONLogger(object):
    def __init__(self, file):
        self.file = file

    def writekvs(self, kvs):
        for k,v in kvs.items():
            if hasattr(v, 'dtype'):
                v = v.tolist()
                kvs[k] = float(v)
        self.file.write(json.dumps(kvs) + '\n')
        self.file.flush()


class WrapEnv(object):
  def __init__(self, level, log_path, width=80, height=80, fps=60):
    self.env = deepmind_lab.Lab(
      level, ['RGB_INTERLACED'],
      config={
          'fps': str(fps),
          'width': str(width),
          'height': str(height)
    })

    self.action_space = len(ACTION_LIST)
    self.observation_space = (80, 80, 1)

    file = open(log_path, "wt")
    self.logger = JSONLogger(file)
    self.rewards = 0
    self.ep_len = 0
    self.action_stat = [0] * self.action_space
    self.tstart = time.time()

  def reset(self):
    self.rewards = 0
    self.ep_len = 0
    self.action_stat = [0] * self.action_space

    self.env.reset()
    obs = self.env.observations()
    obs = obs['RGB_INTERLACED']
    obs = rgb2gray(obs)
    obs = np.expand_dims(obs, axis=2)

    return obs

  def step(self, action):
    reward = self.env.step(ACTION_LIST[action], num_steps=1)
    done = not self.env.is_running()

    if not done:
      obs = self.env.observations()
      obs = obs['RGB_INTERLACED']
      obs = rgb2gray(obs)
      obs = np.expand_dims(obs, axis=2)
    else:
      obs = np.zeros((80, 80, 1), dtype=np.uint8) # todo check datatype, maybe int8

    self.rewards += reward
    self.ep_len += 1
    self.action_stat[action] += 1

    if done:
      epinfo = {"r": self.rewards, "l": self.ep_len, "t": round(time.time() - self.tstart, 6), 'stat': self.action_stat}
      self.logger.writekvs(epinfo)

    return obs, np.clip(reward, -1.0, 1.0), done, {'ale.lives': 0}

  def close(self):
    self.env.close()