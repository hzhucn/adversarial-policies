"""Uses PPO to training an attack policy against a fixed victim policy."""

import functools
import json
import logging
import os
import os.path as osp
import pkgutil

from gym.spaces import Box
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from stable_baselines import GAIL, PPO1, PPO2, SAC
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.gail.dataset.dataset import ExpertDataset
import tensorflow as tf

from aprl.envs.multi_agent import (CurryVecEnv, FlattenSingletonVecEnv, MergeAgentVecEnv,
                                   VecMultiWrapper, make_dummy_vec_multi_env,
                                   make_subproc_vec_multi_env)
from modelfree.common import utils
from modelfree.common.policy_loader import load_policy
from modelfree.envs.gym_compete import (GameOutcomeMonitor, GymCompeteToOurs,
                                        get_policy_type_for_zoo_agent, load_zoo_agent_params)
from modelfree.training.logger import setup_logger
from modelfree.training.scheduling import ConstantAnnealer, Scheduler
from modelfree.training.shaping_wrappers import apply_reward_wrapper, apply_victim_wrapper

train_ex = Experiment('train')
pylog = logging.getLogger('modelfree.train')


class EmbedVictimWrapper(VecMultiWrapper):
    def __init__(self, multi_env, victim, victim_index):
        self.victim = victim
        curried_env = CurryVecEnv(multi_env, self.victim, agent_idx=victim_index)

        super().__init__(curried_env)

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()

    def close(self):
        self.victim.sess.close()
        super().close()


def _save(model, root_dir, save_callbacks):
    os.makedirs(root_dir, exist_ok=True)
    model_path = osp.join(root_dir, 'model.pkl')
    model.save(model_path)
    for f in save_callbacks:
        f(root_dir)


@train_ex.capture
def old_ppo2(_seed, env, out_dir, total_timesteps, num_env, policy,
             batch_size, load_policy, learning_rate, rl_args,
             logger, log_callbacks, save_callbacks):
    try:
        from baselines.ppo2 import ppo2 as ppo2_old
        from baselines import logger as logger_old
    except ImportError as e:
        msg = "{}. HINT: you need to install (OpenAI) Baselines to use old_ppo2".format(e)
        raise ImportError(msg)

    pylog.warning("'old_ppo2' is deprecated; use 'ppo2' where possible. "
                  "Logging and save callbacks not supported amongst other features.")
    logger_old.configure(os.path.join(out_dir, 'old_rl'))

    NETWORK_MAP = {
        'MlpPolicy': 'mlp',
        'MlpLstmPolicy': 'lstm',
        'CnnPolicy': 'cnn',
        'CnnLstmPolilcy': 'cnn_lstm',
    }
    network = NETWORK_MAP[policy]

    graph = tf.Graph()
    sess = utils.make_session(graph)
    load_path = load_policy['path']
    if load_path is not None:
        assert load_policy['type'] == 'old_ppo2'
    with graph.as_default():
        with sess.as_default():
            model = ppo2_old.learn(network=network, env=env,
                                   nsteps=batch_size // num_env,
                                   total_timesteps=total_timesteps,
                                   load_path=load_path,
                                   lr=learning_rate, seed=_seed, **rl_args)

            final_path = osp.join(out_dir, 'final_model')
            _save(model, final_path, save_callbacks)

    return final_path


@train_ex.capture
def _stable(cls, our_type, callback_key, callback_mul, _seed, env, env_name, out_dir,
            total_timesteps, policy, load_policy, rl_args, victim_index, debug, logger,
            log_callbacks, save_callbacks, log_interval, checkpoint_interval, **kwargs):
    kwargs = dict(env=env,
                  verbose=1 if not debug else 2,
                  **kwargs,
                  **rl_args)
    if load_policy['path'] is not None:
        if load_policy['type'] == our_type:
            # SOMEDAY: Counterintuitively this inherits any extra arguments saved in the policy
            model = cls.load(load_policy['path'], **kwargs)
        elif load_policy['type'] == 'zoo':
            policy_cls, policy_kwargs = get_policy_type_for_zoo_agent(env_name)
            kwargs['policy_kwargs'] = policy_kwargs
            model = cls(policy=policy_cls, **kwargs)

            our_idx = 1 - victim_index  # TODO: code duplication?
            params = load_zoo_agent_params(load_policy['path'], env_name, our_idx)
            # We do not need to restore train_model, since it shares params with act_model
            model.act_model.restore(params)
    else:
        model = cls(policy=policy, **kwargs)

    last_checkpoint = 0
    last_log = 0

    def callback(locals, globals):
        nonlocal last_checkpoint, last_log
        step = locals[callback_key] * callback_mul
        if step - checkpoint_interval > last_checkpoint:
            checkpoint_dir = osp.join(out_dir, 'checkpoint', f'{step:012}')
            _save(model, checkpoint_dir, save_callbacks)
            last_checkpoint = step

        if step - log_interval > last_log:
            for f in log_callbacks:
                f(logger, locals, globals)
            last_log = step

        return True  # keep training

    model.learn(total_timesteps=total_timesteps, log_interval=1, seed=_seed, callback=callback)
    final_path = osp.join(out_dir, 'final_model')
    _save(model, final_path, save_callbacks)
    model.sess.close()
    return final_path


def _get_mpi_num_proc():
    # SOMEDAY: If we end up using MPI-based algorithms regularly, come up with a cleaner solution.
    from mpi4py import MPI
    if MPI is None:
        num_proc = 1
    else:
        num_proc = MPI.COMM_WORLD.Get_size()
    return num_proc


class ExpertDatasetFromOurFormat(ExpertDataset):
    """GAIL Expert Dataset. Loads in our format, rather than the GAIL default.

    In particular, GAIL expects a dict of flattened arrays, with episodes concatenated together.
    The episode start is delineated by an `episode_starts` array. See `ExpertDataset` base class
    for more information.

    By contrast, our format consists of a list of NumPy arrays, one for each episode."""
    def __init__(self, expert_path, **kwargs):
        traj_data = np.load(expert_path, allow_pickle=True)

        # Add in episode starts
        episode_starts = []
        for reward_dict in traj_data['rewards']:
            ep_len = len(reward_dict)
            # used to index episodes since they are flattened in GAIL format.
            ep_starts = [True] + [False] * (ep_len - 1)
            episode_starts.append(np.array(ep_starts))

        # Flatten arrays
        traj_data = {k: np.concatenate(v) for k, v in traj_data.items()}
        traj_data['episode_starts'] = np.concatenate(episode_starts)

        # Rename observations->obs
        traj_data['obs'] = traj_data['observations']
        del traj_data['observations']

        super().__init__(traj_data=traj_data, **kwargs)


@train_ex.capture
def gail(batch_size, learning_rate, expert_dataset_path, **kwargs):
    num_proc = _get_mpi_num_proc()
    if expert_dataset_path is None:
        raise ValueError("Must set expert_dataset_path to use GAIL.")
    expert_dataset = ExpertDatasetFromOurFormat(expert_dataset_path)
    kwargs['d_stepsize'] = learning_rate(1)
    kwargs['vf_stepsize'] = learning_rate(1)
    return _stable(GAIL, our_type='gail', expert_dataset=expert_dataset,
                   callback_key='timesteps_so_far', callback_mul=1,
                   timesteps_per_batch=batch_size // num_proc, **kwargs)


@train_ex.capture
def ppo1(batch_size, learning_rate, **kwargs):
    num_proc = _get_mpi_num_proc()
    pylog.warning('Assuming constant learning rate schedule for PPO1')
    optim_stepsize = learning_rate(1)  # PPO1 does not support a callable learning_rate
    return _stable(PPO1, our_type='ppo1', callback_key='timesteps_so_far',
                   callback_mul=batch_size, timesteps_per_actorbatch=batch_size // num_proc,
                   optim_stepsize=optim_stepsize, schedule='constant', **kwargs)


@train_ex.capture
def ppo2(batch_size, num_env, learning_rate, **kwargs):
    return _stable(PPO2, our_type='ppo2', callback_key='update', callback_mul=batch_size,
                   n_steps=batch_size // num_env, learning_rate=learning_rate, **kwargs)


@train_ex.capture
def sac(batch_size, learning_rate, **kwargs):
    return _stable(SAC, our_type='sac', callback_key='step', callback_mul=1,
                   batch_size=batch_size, learning_rate=learning_rate, **kwargs)


@train_ex.config
def train_config():
    # Logging
    root_dir = "data/baselines"     # root of directory to store baselines log
    exp_name = "default"            # name of experiment

    # Environment
    env_name = "multicomp/SumoAnts-v0"  # Gym environment ID
    num_env = 8                     # number of environments to run in parallel
    total_timesteps = 4096          # total number of timesteps to training for

    # Victim Config
    victim_type = "zoo"             # type supported by policy_loader.py
    victim_path = "1"               # path or other unique identifier
    victim_index = 0                # which agent the victim is (we default to other agent)

    # RL Algorithm Hyperparameters
    rl_algo = "ppo2"                # RL algorithm to use
    policy = "MlpPolicy"            # policy network type
    batch_size = 2048               # batch size
    learning_rate = 3e-4            # learning rate
    normalize = True                # normalize environment observations and reward
    rl_args = dict()                # algorithm-specific arguments
    adv_noise_params = None         # param dict for epsilon-ball noise policy added to zoo policy

    # RL Algorithm Policies/Demonstrations
    load_policy = {                 # fine-tune this policy
        'path': None,               # path with policy weights
        'type': rl_algo,            # type supported by policy_loader.py
    }
    expert_dataset_path = None      # path to trajectory data to train GAIL

    # General
    checkpoint_interval = 131072    # save weights to disk after this many timesteps
    log_interval = 2048             # log statistics to disk after this many timesteps
    log_output_formats = None       # custom output formats for logging
    debug = False                   # debug mode; may run more slowly
    seed = 0                        # random seed
    _ = locals()  # quieten flake8 unused variable warning
    del _


DEFAULT_CONFIGS = {}


def load_default(env_name, config_dir):
    default_config = DEFAULT_CONFIGS.get(env_name, 'default.json')
    fname = os.path.join('configs', config_dir, default_config)
    config = pkgutil.get_data('modelfree', fname)
    return json.loads(config)


@train_ex.config
def wrappers_config(env_name):
    rew_shape = True  # enable reward shaping
    rew_shape_params = load_default(env_name, 'rew')  # parameters for reward shaping

    victim_noise = False  # enable adding noise to victim
    victim_noise_params = load_default(env_name, 'noise')  # parameters for victim noise

    _ = locals()  # quieten flake8 unused variable warning
    del _


@train_ex.capture
def build_env(out_dir, _seed, env_name, num_env, victim_type, victim_index, debug):
    pre_wrapper = GymCompeteToOurs if env_name.startswith('multicomp/') else None

    if victim_type == 'none':
        our_idx = 0
    else:
        our_idx = 1 - victim_index

    def env_fn(i):
        return utils.make_env(env_name, _seed, i, out_dir, our_idx, pre_wrapper=pre_wrapper)

    if not debug and num_env > 1:
        make_vec_env = make_subproc_vec_multi_env
    else:
        make_vec_env = make_dummy_vec_multi_env
    multi_venv = make_vec_env([functools.partial(env_fn, i) for i in range(num_env)])

    if victim_type == 'none':
        assert multi_venv.num_agents == 1, "No victim only works in single-agent environments"
    else:
        assert multi_venv.num_agents == 2, "Need two-agent environment when victim"

    return multi_venv, our_idx


@train_ex.capture
def multi_wrappers(multi_venv, env_name, log_callbacks):
    if env_name.startswith('multicomp/'):
        game_outcome = GameOutcomeMonitor(multi_venv)
        # Need game_outcome as separate variable as Python closures bind late
        log_callbacks.append(lambda logger, locals, globals: game_outcome.log_callback(logger))
        multi_venv = game_outcome

    return multi_venv


@train_ex.capture
def wrap_adv_noise_ball(env_name, our_idx, multi_venv, adv_noise_params, victim_path, victim_type):
    adv_noise_agent_val = adv_noise_params['noise_val']
    base_policy_path = adv_noise_params.get('base_path', victim_path)
    base_policy_type = adv_noise_params.get('base_type', victim_type)
    base_policy = load_policy(policy_path=base_policy_path, policy_type=base_policy_type,
                              env=multi_venv, env_name=env_name, index=our_idx)
    base_action_space = multi_venv.action_space.spaces[our_idx]
    adv_noise_action_space = Box(low=adv_noise_agent_val * base_action_space.low,
                                 high=adv_noise_agent_val * base_action_space.high)
    multi_venv = MergeAgentVecEnv(venv=multi_venv, policy=base_policy,
                                  replace_action_space=adv_noise_action_space,
                                  merge_agent_idx=our_idx)
    return multi_venv


@train_ex.capture
def maybe_embed_victim(multi_venv, our_idx, scheduler, log_callbacks, env_name, victim_type,
                       victim_path, victim_index, victim_noise, victim_noise_params,
                       adv_noise_params):
    if victim_type != 'none':
        # If we are actually training an epsilon-ball noise agent on top of a zoo agent
        if adv_noise_params is not None:
            multi_venv = wrap_adv_noise_ball(env_name, our_idx, multi_venv)

        # Load the victim and then wrap it if appropriate.
        victim = load_policy(policy_path=victim_path, policy_type=victim_type, env=multi_venv,
                             env_name=env_name, index=victim_index)
        if victim_noise:
            victim = apply_victim_wrapper(victim=victim, noise_params=victim_noise_params,
                                          scheduler=scheduler)
            log_callbacks.append(lambda logger, locals, globals: victim.log_callback(logger))

        # Curry the victim
        multi_venv = EmbedVictimWrapper(multi_env=multi_venv, victim=victim,
                                        victim_index=victim_index)

    return multi_venv


@train_ex.capture
def single_wrappers(single_venv, scheduler, our_idx, normalize, load_policy,
                    rew_shape, rew_shape_params, log_callbacks, save_callbacks):
    if rew_shape:
        rew_shape_venv = apply_reward_wrapper(single_env=single_venv, scheduler=scheduler,
                                              shaping_params=rew_shape_params, agent_idx=our_idx)
        log_callbacks.append(lambda logger, locals, globals: rew_shape_venv.log_callback(logger))
        single_venv = rew_shape_venv

        for anneal_type in ['noise', 'rew_shape']:
            if scheduler.is_conditional(anneal_type):
                scheduler.set_annealer_get_logs(anneal_type, rew_shape_venv.get_logs)

    if normalize:
        if load_policy['type'] == 'zoo':
            raise ValueError("Trying to normalize twice. Bansal et al's Zoo agents normalize "
                             "implicitly. Please set normalize=False to disable VecNormalize.")

        normalized_venv = VecNormalize(single_venv)
        save_callbacks.append(lambda root_dir: normalized_venv.save_running_average(root_dir))
        single_venv = normalized_venv

    return single_venv


RL_ALGOS = {
    'gail': gail,
    'ppo1': ppo1,
    'ppo2': ppo2,
    'old_ppo2': old_ppo2,
    'sac': sac,
}


# True for Stable Baselines as of 2019-03
NO_VECENV = ['ddpg', 'dqn', 'gail', 'her', 'ppo1', 'sac']


@train_ex.main
def train(_run, root_dir, exp_name, num_env, rl_algo, learning_rate, log_output_formats):
    scheduler = Scheduler(annealer_dict={'lr': ConstantAnnealer(learning_rate)})
    out_dir, logger = setup_logger(root_dir, exp_name, output_formats=log_output_formats)
    log_callbacks, save_callbacks = [], []
    pylog.info(f"Log output formats: {logger.output_formats}")

    if rl_algo in NO_VECENV and num_env > 1:
        raise ValueError(f"'{rl_algo}' needs 'num_env' set to 1.")

    multi_venv, our_idx = build_env(out_dir)
    multi_venv = multi_wrappers(multi_venv, log_callbacks=log_callbacks)
    multi_venv = maybe_embed_victim(multi_venv, our_idx, scheduler, log_callbacks=log_callbacks)

    single_venv = FlattenSingletonVecEnv(multi_venv)
    single_venv = single_wrappers(single_venv, scheduler, our_idx,
                                  log_callbacks=log_callbacks, save_callbacks=save_callbacks)

    train_fn = RL_ALGOS[rl_algo]
    res = train_fn(env=single_venv, out_dir=out_dir, learning_rate=scheduler.get_annealer('lr'),
                   logger=logger, log_callbacks=log_callbacks, save_callbacks=save_callbacks)
    single_venv.close()

    return res


def main():
    observer = FileStorageObserver.create(osp.join('data', 'sacred', 'train'))
    train_ex.observers.append(observer)
    train_ex.run_commandline()


if __name__ == '__main__':
    main()
