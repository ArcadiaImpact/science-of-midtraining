"""Offline consistency checks for the final selected-cell handoff."""
from collections import Counter
import json
from pathlib import Path


def test_final_handoff_has_no_sender_work_or_ownership_holds():
    root = Path(__file__).resolve().parents[1]
    path = root / 'experiments/dispatch/dispatch_final_v1/JONATHAN_GEMMA_HALFPCT_CELLS.json'
    plan = json.loads(path.read_text())
    cells = plan['cells']
    assert len(cells) == len({c['id'] for c in cells}) == 36
    assert Counter(c['action'] for c in cells) == {
        'skip_complete': 18, 'jonathan_train_eval': 14, 'jonathan_eval_only': 4}
    assert plan['sender_work_complete'] and plan['sender_results_pending'] == 0
    assert not any(c['ownership_confirm_before_start'] for c in cells)
    released = [c for c in cells if c.get('sender_ownership_released')]
    assert len(released) == 5
    assert all(c['action'] == 'jonathan_train_eval' and c['prior_attempt_may_exist'] for c in released)
    for cell in cells:
        if cell['action'] == 'skip_complete':
            assert cell['complete_on_hf']
            assert cell['checkpoints'] == plan['recipe']['saves']
            assert cell['eval_steps_complete'] == [256, 512]
        elif cell['action'] == 'jonathan_eval_only':
            assert 512 in cell['checkpoints']
            assert cell['eval_steps_complete'] == [256]
    assert sum('completion_evidence' in c for c in cells) == 9
