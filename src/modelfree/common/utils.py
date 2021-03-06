from collections import defaultdict
import datetime
import itertools
import os
from os import path as osp
import warnings

import gym
from gym import Wrapper
from gym.monitoring import VideoRecorder
import numpy as np
from stable_baselines.common import BaseRLModel
from stable_baselines.common.policies import BasePolicy
import tensorflow as tf

from aprl.common.multi_monitor import MultiMonitor
from aprl.envs.multi_agent import MultiAgentEnv, SingleToMulti, VecMultiWrapper


class DummyModel(BaseRLModel):
    """Abstract class for policies pretending to be RL algorithms (models).

    Provides stub implementations that raise NotImplementedError.
    The predict method is left as abstract and must be implemented in base class."""
    def __init__(self, policy, sess):
        """Constructs a DummyModel with given policy and session.
        :param policy: (BasePolicy) a loaded policy.
        :param sess: (tf.Session or None) a TensorFlow session.
        :return an instance of BaseRLModel.
        """
        super().__init__(policy=policy, env=None, requires_vec_env=True, policy_base='Dummy')
        self.sess = sess

    def setup_model(self):
        raise NotImplementedError()

    def learn(self):
        raise NotImplementedError()

    def action_probability(self, observation, state=None, mask=None, actions=None):
        raise NotImplementedError()

    def save(self, save_path):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def _get_pretrain_placeholders(self):
        raise NotImplementedError()


class PolicyToModel(DummyModel):
    """Converts BasePolicy to a BaseRLModel with only predict implemented."""
    def __init__(self, policy):
        """Constructs a BaseRLModel using policy for predictions.
        :param policy: (BasePolicy) a loaded policy.
        :return an instance of BaseRLModel.
        """
        super().__init__(policy=policy, sess=policy.sess)

    def predict(self, observation, state=None, mask=None, deterministic=False):
        if state is None:
            state = self.policy.initial_state
        if mask is None:
            mask = [False for _ in range(self.policy.n_env)]

        actions, _val, states, _neglogp = self.policy.step(observation, state, mask,
                                                           deterministic=deterministic)
        return actions, states


class OpenAIToStablePolicy(BasePolicy):
    """Converts an OpenAI Baselines Policy to a Stable Baselines policy."""
    def __init__(self, old_policy):
        self.old = old_policy
        self.sess = old_policy.sess

    @property
    def initial_state(self):
        return self.old.initial_state

    def step(self, obs, state=None, mask=None, deterministic=False):
        stochastic = not deterministic
        return self.old.step(obs, S=state, M=mask, stochastic=stochastic)

    def proba_step(self, obs, state=None, mask=None):
        raise NotImplementedError()


class ConstantPolicy(BasePolicy):
    """Policy that returns a constant action."""
    def __init__(self, env, constant):
        assert env.action_space.contains(constant)
        super().__init__(sess=None,
                         ob_space=env.observation_space,
                         ac_space=env.action_space,
                         n_env=env.num_envs,
                         n_steps=1,
                         n_batch=1)
        self.constant = constant

    def step(self, obs, state=None, mask=None, deterministic=False):
        actions = np.array([self.constant] * self.n_env)
        return actions, None, None, None

    def proba_step(self, obs, state=None, mask=None):
        return self.step(obs, state=state, mask=mask)


class ZeroPolicy(ConstantPolicy):
    """Policy that returns a zero action."""
    def __init__(self, env):
        super().__init__(env, np.zeros(env.action_space.shape))


class RandomPolicy(BasePolicy):
    def __init__(self, env):
        super().__init__(sess=None,
                         ob_space=env.observation_space,
                         ac_space=env.action_space,
                         n_env=env.num_envs,
                         n_steps=1,
                         n_batch=1)

    def step(self, obs, state=None, mask=None, deterministic=False):
        actions = np.array([self.ac_space.sample() for _ in range(self.n_env)])
        return actions, None, None, None

    def proba_step(self, obs, state=None, mask=None):
        raise NotImplementedError()


class VideoWrapper(Wrapper):
    def __init__(self, env, directory):
        super(VideoWrapper, self).__init__(env)
        self.directory = osp.abspath(directory)
        os.makedirs(self.directory, exist_ok=True)
        self.episode_id = 0
        self.video_recorder = None

    def _step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done:
            winners = [i for i, d in info.items() if 'winner' in d]
            metadata = {'winners': winners}
            self._reset_video_recorder(metadata)
        self.video_recorder.capture_frame()
        return obs, rew, done, info

    def _reset(self):
        self._reset_video_recorder()
        self.episode_id += 1
        return self.env.reset()

    def _reset_video_recorder(self, metadata=None):
        if self.video_recorder:
            if metadata is not None:
                self.video_recorder.metadata.update(metadata)
            self.video_recorder.close()
        self.video_recorder = VideoRecorder(
            env=self.env,
            base_path=osp.join(self.directory, 'video.{:06}'.format(self.episode_id)),
            metadata={'episode_id': self.episode_id},
        )


def make_session(graph=None):
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(graph=graph, config=tf_config)
    return sess


def _filter_dict(d, keys):
    """Filter a dictionary to contain only the specified keys.

    If keys is None, it returns the dictionary verbatim.
    If a key in keys is not present in the dictionary, it gives a warning, but does not fail.

    :param d: (dict)
    :param keys: (iterable) the desired set of keys; if None, performs no filtering.
    :return (dict) a filtered dictionary."""
    if keys is None:
        return d
    else:
        keys = set(keys)
        present_keys = keys.intersect(d.keys())
        missing_keys = keys.difference(d.keys())
        res = {k: d[k] for k in present_keys}
        if missing_keys is not None:
            warnings.warn("Missing expected keys: {}".format(missing_keys), stacklevel=2)
        return res


class TrajectoryRecorder(VecMultiWrapper):
    """Class for recording and saving trajectories in numpy.npz format.
    For each episode, we record observations, actions, rewards and optionally network activations
    for the agents specified by agent_indices.

    :param venv: (VecEnv) environment to wrap
    :param agent_indices: (list,int) indices of agents whose trajectories to record
    :param env_keys: (list,str) keys for environment data to record; if None, record all.
                     Options are 'observations', 'actions' and 'rewards'.
    :param info_keys: (list,str) keys in the info dict to record; if None, record all.
                      This is often used to expose activations from the policy.
    """

    def __init__(self, venv, agent_indices=None, env_keys=None, info_keys=None):
        super().__init__(venv)

        if agent_indices is None:
            self.agent_indices = range(self.num_agents)
        elif isinstance(agent_indices, int):
            self.agent_indices = [agent_indices]
        self.env_keys = env_keys
        self.info_keys = info_keys

        self.traj_dicts = [[defaultdict(list) for _ in range(self.num_envs)]
                           for _ in self.agent_indices]
        self.full_traj_dicts = [defaultdict(list) for _ in self.agent_indices]
        self.prev_obs = None
        self.actions = None

    def step_async(self, actions):
        self.actions = actions
        self.venv.step_async(actions)

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        self.record_timestep_data(self.prev_obs, self.actions, rewards, dones, infos)
        self.prev_obs = observations
        return observations, rewards, dones, infos

    def reset(self):
        observations = self.venv.reset()
        self.prev_obs = observations
        return observations

    def record_timestep_data(self, prev_obs, actions, rewards, dones, infos):
        """Record observations, actions, rewards, and (optionally) network activations
        of one timestep in dict for current episode. Completed episode trajectories are
        collected in a list in preparation for being saved to disk.

        :param prev_obs: (np.ndarray<float>) observations from previous timestep
        :param actions: (np.ndarray<float>) actions taken after observing prev_obs
        :param rewards: (np.ndarray<float>) rewards from actions
        :param dones: ([bool]) whether episode ended (not recorded)
        :param infos: ([dict]) dicts with network activations if networks are transparent
        :return: None
        """
        env_data = {
            'observations': prev_obs,
            'actions': actions,
            'rewards': rewards,
        }
        env_data = _filter_dict(env_data, self.env_keys)

        iter_space = itertools.product(enumerate(self.traj_dicts), range(self.num_envs))
        # iterate over both agents over all environments in VecEnv
        for (dict_idx, agent_dicts), env_idx in iter_space:
            # in dict number dict_idx, record trajectories for agent number agent_idx
            agent_idx = self.agent_indices[dict_idx]
            for key, val in env_data.items():
                # data_vals always have data for all agents (use agent_idx not dict_idx)
                agent_dicts[env_idx][key].append(val[agent_idx][env_idx])

            info_dict = infos[env_idx][agent_idx]
            info_dict = _filter_dict(info_dict, self.info_keys)
            for key, val in info_dict.items():
                agent_dicts[env_idx][key].append(val)

            if dones[env_idx]:
                ep_ret = sum(agent_dicts[env_idx]['rewards'])
                self.full_traj_dicts[dict_idx]['episode_returns'].append(np.array([ep_ret]))

                for key, val in agent_dicts[env_idx].items():
                    # consolidate episode data and append to long-term data dict
                    episode_key_data = np.array(val)
                    self.full_traj_dicts[dict_idx][key].append(episode_key_data)
                agent_dicts[env_idx] = defaultdict(list)

    def save(self, save_dir):
        """Save trajectories to save_dir in NumPy compressed-array format, per-agent.

        Our format consists of a dictionary with keys -- e.g. 'observations', 'actions'
        and 'rewards' -- containing lists of NumPy arrays, one for each episode.

        :param save_dir: (str) path to save trajectories; will create directory if needed.
        :return None
        """
        os.makedirs(save_dir, exist_ok=True)
        for dict_idx, agent_idx in enumerate(self.agent_indices):
            agent_dicts = self.full_traj_dicts[dict_idx]
            dump_dict = {k: np.asarray(v) for k, v in agent_dicts.items()}

            save_path = os.path.join(save_dir, f'agent_{agent_idx}.npz')
            np.savez(save_path, **dump_dict)


def simulate(venv, policies, render=False):
    """
    Run Environment env with the policies in `policies`.
    :param venv(VecEnv): vector environment.
    :param policies(list<BaseModel>): a policy per agent.
    :param render: (bool) true if the run should be rendered to the screen
    :return: streams information about the simulation
    """
    observations = venv.reset()
    dones = [False] * venv.num_envs
    states = [None for _ in policies]

    while True:
        if render:
            venv.render()

        actions = []
        new_states = []
        for policy, obs, state in zip(policies, observations, states):
            act, new_state = policy.predict(obs, state=state, mask=dones)
            actions.append(act)
            new_states.append(new_state)
        actions = tuple(actions)
        states = new_states

        observations, rewards, dones, infos = venv.step(actions)
        yield observations, rewards, dones, infos


def make_env(env_name, seed, i, out_dir, our_idx=None, pre_wrapper=None, post_wrapper=None):
    multi_env = gym.make(env_name)
    if pre_wrapper is not None:
        multi_env = pre_wrapper(multi_env)
    if not isinstance(multi_env, MultiAgentEnv):
        multi_env = SingleToMulti(multi_env)
    multi_env.seed(seed + i)

    if out_dir is not None:
        mon_dir = osp.join(out_dir, 'mon')
        os.makedirs(mon_dir, exist_ok=True)
        multi_env = MultiMonitor(multi_env, osp.join(mon_dir, 'log{}'.format(i)), our_idx)

    if post_wrapper is not None:
        multi_env = post_wrapper(multi_env)

    return multi_env


def make_timestamp():
    ISO_TIMESTAMP = "%Y%m%d_%H%M%S"
    return datetime.datetime.now().strftime(ISO_TIMESTAMP)
