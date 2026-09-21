from experiments.dispatch.dispatch_final_v1.ops.prune_handled_notifications import handled


def item(text):
    return {'input': [{'type': 'text', 'text': text}]}


def test_exact_serviced_event_only():
    checks = '## date — event123 serviced\n'
    assert handled(item('User-authorized AFT event notification 123: result'), checks, '123', '')
    assert not handled(item('User-authorized AFT event notification 123: result'), checks, '122', '')
    assert not handled(item('User-authorized AFT event notification 124: result'), checks, '999', '')
    assert not handled(item('Please inspect event123'), checks, '999', '')
    assert not handled(item('User-authorized AFT event notification 123: result'), 'event123 pending', '999', '')


def test_exact_serviced_heartbeat_only():
    stamp = '2026-09-08T03:55:32+00:00'
    message = item('User-authorized 15m GLM + Gemma AFT heartbeat '+stamp+'. Inspect')
    checks = '## date — heartbeat'+stamp+' serviced\n'
    assert handled(message, checks, '123', stamp)
    assert not handled(message, checks, '123', '2026-09-08T03:40:32+00:00')
    assert not handled(message, '', '123', stamp)


def test_preserve_extra_input():
    message = item('User-authorized AFT event notification 123: result')
    message['input'].append({'type': 'text', 'text': 'Unrelated user instruction'})
    assert not handled(message, '## date — event123 serviced\n', '123', '')
