"""Named configs for modelfree.multi.score."""

import itertools

from ray import tune

from modelfree.configs.multi.common import BANSAL_GOOD_ENVS, VICTIM_INDEX
from modelfree.envs import gym_compete


def _env_agents(envs=None):
    if envs is None:
        envs = BANSAL_GOOD_ENVS
    env_and_agents = []
    for env in envs:
        num_zoo = gym_compete.num_zoo_policies(env)
        zoo_ids = itertools.product(range(num_zoo), range(num_zoo))
        env_and_agents += [(env, i + 1, j + 1) for (i, j) in zoo_ids]
    return env_and_agents


def _fixed_vs_victim(fixed_type, envs=None):
    if envs is None:
        envs = BANSAL_GOOD_ENVS
    results = []
    for env in envs:
        num_zoo = gym_compete.num_zoo_policies(env)
        victim_index = VICTIM_INDEX[env]
        if victim_index == 0:
            results += [(env, 'zoo', i + 1, fixed_type, 'none') for i in range(num_zoo)]
        elif victim_index == 1:
            results += [(env, fixed_type, 'none', 'zoo', i + 1) for i in range(num_zoo)]
        else:
            raise ValueError(f"Victim index '{victim_index}' out of range")
    return results


def _high_accuracy(score):
    score['episodes'] = 1000
    score['num_env'] = 16


PATHS_ONLY = 'env_name:agent_a_path:agent_b_path'
PATHS_AND_TYPES = 'env_name:agent_a_type:agent_a_path:agent_b_type:agent_b_path'


def make_configs(multi_score_ex):
    @multi_score_ex.named_config
    def zoo_baseline(score):
        """Try all pre-trained policies from Bansal et al's gym_compete zoo against each other."""
        score = dict(score)
        _high_accuracy(score)
        score['agent_a_type'] = 'zoo'
        score['agent_b_type'] = 'zoo'
        spec = {
            'config': {
                PATHS_ONLY: tune.grid_search(_env_agents()),
            },
        }
        exp_name = 'zoo_baseline'
        _ = locals()  # quieten flake8 unused variable warning
        del _

    @multi_score_ex.named_config
    def fixed_baseline(score):
        """Try zero-agent and random-agent against pre-trained zoo policies."""
        score = dict(score)
        _high_accuracy(score)
        spec = {
            'config': {
                PATHS_AND_TYPES: tune.grid_search(_fixed_vs_victim('random') +
                                                  _fixed_vs_victim('zero')),
            }
        }
        exp_name = 'fixed_baseline'
        _ = locals()  # quieten flake8 unused variable warning
        del _
