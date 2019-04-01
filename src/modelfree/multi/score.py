"""Hyperparameter search for train.py using Ray Tune."""

import logging
import os.path as osp

from sacred import Experiment
from sacred.observers import FileStorageObserver

from modelfree.multi import common
from modelfree.multi.score_worker import score_worker
from modelfree.score_agent import score_ex

multi_score_ex = Experiment('multi_score', ingredients=[score_ex])
pylog = logging.getLogger('modelfree.multi_score')

# Load common configs (e.g. upload directories) and define the run command
run = common.make_sacred(multi_score_ex, 'score', score_worker)

# Load named configs for individual experiments (these change a lot, so keep out of this file)
# TODO: named configs
# TODO: add test case


@multi_score_ex.config
def default_config(score):
    spec = {  # experiment specification
        'run': 'score',
        # TODO: tune number of actual CPUs required
        'resources_per_trial': {'cpu': score['num_env'] // 2},
    }

    _ = locals()  # quieten flake8 unused variable warning
    del _


@multi_score_ex.main
def multi_score(score):
    run(base_config=score)


def main():
    observer = FileStorageObserver.create(osp.join('data', 'sacred', 'multi_score'))
    multi_score_ex.observers.append(observer)
    multi_score_ex.run_commandline()


if __name__ == '__main__':
    main()
