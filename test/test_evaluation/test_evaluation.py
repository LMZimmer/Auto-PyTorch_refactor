import os
import logging
import shutil
import sys
import time
import unittest
import unittest.mock

import numpy as np
import pynisher
from smac.runhistory.runhistory import RunInfo
from smac.stats.stats import Stats
from smac.tae import StatusType
from smac.utils.constants import MAXINT

from autoPyTorch.evaluation.tae import ExecuteTaFuncWithQueue, get_cost_of_crash
from autoPyTorch.pipeline.components.training.metrics.metrics import accuracy, log_loss

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)
from evaluation_util import get_multiclass_classification_datamanager  # noqa E402


def safe_eval_success_mock(*args, **kwargs):
    queue = kwargs['queue']
    queue.put({'status': StatusType.SUCCESS,
               'loss': 0.5,
               'additional_run_info': ''})


class BackendMock(object):
    def __init__(self):
        self.temporary_directory = './.tmp_evaluation'
        try:
            os.mkdir(self.temporary_directory)
        except:  # noqa 3722
            pass
    def load_datamanager(self):
        return get_multiclass_classification_datamanager()


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.datamanager = get_multiclass_classification_datamanager()
        self.tmp = os.path.join(os.getcwd(), '.test_evaluation')
        os.mkdir(self.tmp)
        self.logger = logging.getLogger()
        scenario_mock = unittest.mock.Mock()
        scenario_mock.wallclock_limit = 10
        scenario_mock.algo_runs_timelimit = 1000
        scenario_mock.ta_run_limit = 100
        self.scenario = scenario_mock
        stats = Stats(scenario_mock)
        stats.start_timing()
        self.stats = stats

        try:
            shutil.rmtree(self.tmp)
        except Exception:
            pass

    def tearDown(self):
        try:
            shutil.rmtree(self.tmp)
        except Exception:
            pass

    ############################################################################
    # pynisher tests
    def test_pynisher_memory_error(self):
        def fill_memory():
            a = np.random.random_sample((10000, 10000)).astype(np.float64)
            return np.sum(a)

        safe_eval = pynisher.enforce_limits(mem_in_mb=1)(fill_memory)
        safe_eval()
        self.assertEqual(safe_eval.exit_status, pynisher.MemorylimitException)

    def test_pynisher_timeout(self):
        def run_over_time():
            time.sleep(2)

        safe_eval = pynisher.enforce_limits(wall_time_in_s=1,
                                            grace_period_in_s=0)(run_over_time)
        safe_eval()
        self.assertEqual(safe_eval.exit_status, pynisher.TimeoutException)

    ############################################################################
    # Test ExecuteTaFuncWithQueue.run_wrapper()
    @unittest.mock.patch('pynisher.enforce_limits')
    def test_cutoff_lower_than_remaining_time(self, pynisher_mock):
        config = unittest.mock.Mock()
        config.config_id = 198
        ta = ExecuteTaFuncWithQueue(backend=BackendMock(), seed=1,
                                    stats=self.stats,
                                    memory_limit=3072,
                                    metric=accuracy,
                                    cost_for_crash=get_cost_of_crash(accuracy),
                                    abort_on_first_run_crash=False,
                                    logger=self.logger
                                    )
        self.stats.ta_runs = 1
        ta.run_wrapper(RunInfo(config=config, cutoff=30, instance=None, instance_specific=None,
                               seed=1, capped=False))
        self.assertEqual(pynisher_mock.call_args[1]['wall_time_in_s'], 4)
        self.assertIsInstance(pynisher_mock.call_args[1]['wall_time_in_s'], int)

    @unittest.mock.patch('pynisher.enforce_limits')
    def test_eval_with_limits_holdout_fail_timeout(self, pynisher_mock):
        config = unittest.mock.Mock()
        config.config_id = 198

        m1 = unittest.mock.Mock()
        m2 = unittest.mock.Mock()
        m1.return_value = m2
        pynisher_mock.return_value = m1
        m2.exit_status = pynisher.TimeoutException
        m2.wall_clock_time = 30
        ta = ExecuteTaFuncWithQueue(backend=BackendMock(), seed=1,
                                    stats=self.stats,
                                    memory_limit=3072,
                                    metric=accuracy,
                                    cost_for_crash=get_cost_of_crash(accuracy),
                                    abort_on_first_run_crash=False,
                                    logger=self.logger
                                    )
        info = ta.run_wrapper(RunInfo(config=config, cutoff=30, instance=None,
                                      instance_specific=None, seed=1, capped=False))
        self.assertEqual(info[1].status, StatusType.TIMEOUT)
        self.assertEqual(info[1].cost, 1.0)
        self.assertIsInstance(info[1].time, float)
        self.assertNotIn('exitcode', info[1].additional_info)
