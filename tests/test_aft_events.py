import unittest
from experiments.prior_coins.dispatch_final_v1.ops.aft_events import detect


def data(step=1, stage='train', ok=True, done=0):
    return {'age_seconds': 5, 'handruns': [
        {'label': 'gemma-grid/test', 'running': True, 'probe':
         {'ok': ok, 'stage': stage, 'step': step, 'total': 512, 'cells_done': done}}]}


class Events(unittest.TestCase):
    def test_null_setup_count_is_safe(self):
        s = {}
        self.assertEqual(detect(data(done=None), s, 0), [])
        self.assertEqual(detect(data(done=None), s, 30), [])
        self.assertEqual(s['workers']['gemma-grid/test']['cells_done'], 0)
        self.assertTrue(detect(data(done=1), s, 60))

    def test_old_null_count_is_safe(self):
        s = {'workers': {'gemma-grid/test': {'cells_done': None}}}
        self.assertEqual(detect(data(done=None), s, 0), [])

    def test_baseline_and_normal_steps_silent(self):
        s = {}
        self.assertEqual(detect(data(), s, 0), [])
        self.assertEqual(detect(data(2), s, 30), [])

    def test_completion_once(self):
        s = {}; detect(data(), s, 0)
        self.assertTrue(detect(data(512), s, 30))
        self.assertEqual(detect(data(512), s, 60), [])

    def test_stage_and_cell_change(self):
        s = {}; detect(data(), s, 0)
        self.assertEqual(len(detect(data(stage='eval', done=1), s, 30)), 2)

    def test_transient_failure_debounce(self):
        s = {}; detect(data(), s, 0)
        self.assertEqual(detect(data(ok=False), s, 30), [])
        self.assertTrue(detect(data(ok=False), s, 60))
        self.assertEqual(detect(data(ok=False), s, 90), [])

    def test_stall_once_and_recovery(self):
        s = {}; detect(data(), s, 0)
        self.assertTrue(detect(data(), s, 1201))
        self.assertEqual(detect(data(), s, 1231), [])
        self.assertEqual(detect(data(2), s, 1261), [])

    def test_stale_deduplicated(self):
        s = {}; d = {'collector_error': 'offline'}
        self.assertTrue(detect(d, s, 0))
        self.assertEqual(detect(d, s, 30), [])

    def test_epoch1_eval_once(self):
        s = {}; d = data(18, 'eval'); d['handruns'][0]['probe']['total'] = 38
        detect(d, s, 0)
        d['handruns'][0]['probe']['step'] = 19
        self.assertTrue(detect(d, s, 30))
        self.assertEqual(detect(d, s, 60), [])


if __name__ == '__main__':
    unittest.main()
