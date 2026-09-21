"""Remove only AFT notifications with exact serviced-log and acknowledgement evidence.

Uses the installed app-server queue API, never edits Codex databases or starts
turns. Dry-run by default. Full submissions are archived before any deletion.
"""
import argparse
import json
import os
from pathlib import Path
import re
import select
import subprocess
import tempfile
import time

THREAD = '01a07c4a-9049-7f01-ae39-a9a938c969b8'
BASE = Path(__file__).resolve().parents[4] / 'artifacts/aft_size_mixture_v1'


def handled(item, checks, event_ack, heartbeat_ack):
    inputs = item.get('input', [])
    if len(inputs) != 1 or inputs[0].get('type') != 'text':
        return False
    text = inputs[0].get('text', '')
    match = re.match(r'^User-authorized AFT event notification (\d+): ', text)
    if match:
        ident = match[1]
        return (int(ident) <= int(event_ack) and
                re.search(r'^## .* — event' + ident + r' serviced$', checks, re.M) is not None)
    match = re.match(r'^User-authorized 15m GLM \+ Gemma AFT heartbeat ([0-9T:+-]+)\. ', text)
    if match:
        ident = match[1]
        return (ident <= heartbeat_ack and
                re.search(r'^## .* — heartbeat' + re.escape(ident) + r' serviced$', checks, re.M) is not None)
    return False


class QueueClient:
    def __init__(self):
        self.errors = tempfile.TemporaryFile()
        env = {k: v for k, v in os.environ.items() if k not in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY')}
        self.proc = subprocess.Popen(['codex', 'app-server'], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.errors, env=env)
        self.sequence = 0
        self.buffer = b''
        self.rpc('initialize', {'clientInfo': {'name': 'aft_monitor_maintenance', 'version': '1'},
                                'capabilities': {'experimentalApi': True}})
        self.send({'method': 'initialized', 'params': {}})

    def send(self, message):
        self.proc.stdin.write((json.dumps(message) + '\n').encode())
        self.proc.stdin.flush()

    def rpc(self, method, params):
        self.sequence += 1
        ident = self.sequence
        self.send(dict(id=ident, method=method, params=params))
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            while b'\n' in self.buffer:
                line, self.buffer = self.buffer.split(b'\n', 1)
                message = json.loads(line)
                if message.get('id') == ident:
                    if 'error' in message:
                        raise RuntimeError(message['error'])
                    return message['result']
            if not select.select([self.proc.stdout], [], [], max(0, deadline-time.monotonic()))[0]:
                break
            data = os.read(self.proc.stdout.fileno(), 65536)
            if not data:
                raise RuntimeError('Queue API process closed')
            self.buffer += data
        raise TimeoutError('Ambiguous queue API timeout; reconcile before retry')

    def items(self):
        result, cursor = [], None
        while True:
            page = self.rpc('thread/queue/list', dict(threadId=THREAD, limit=100, cursor=cursor))
            result.extend(page['data'])
            cursor = page.get('nextCursor')
            if not cursor:
                return result

    def close(self):
        self.proc.terminate()
        self.proc.wait(timeout=5)
        self.errors.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    checks = (BASE/'heartbeat/checks.md').read_text()
    event_ack = (BASE/'events/ack.txt').read_text().strip()
    heartbeat_ack = (BASE/'heartbeat/ack.txt').read_text().strip()
    client = QueueClient()
    try:
        items = client.items()
        candidates = [x for x in items if handled(x, checks, event_ack, heartbeat_ack)]
        print(json.dumps(dict(queued=len(items),handled_candidates=len(candidates),execute=args.execute)), flush=True)
        if not args.execute or not candidates:
            return
        archive = BASE/'events/handled-queue-archive'/f'{time.time_ns()}.json'
        archive.parent.mkdir(parents=True, exist_ok=True)
        record = dict(thread=THREAD, event_ack=event_ack, heartbeat_ack=heartbeat_ack,
                      submissions=candidates, deleted=[], complete=False)
        def persist():
            temporary = archive.with_suffix('.tmp')
            with temporary.open('w') as stream:
                json.dump(record, stream, indent=2)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(archive)
        persist()
        for item in candidates:
            client.rpc('thread/queue/delete', dict(threadId=THREAD,queuedSubmissionId=item['id']))
            record['deleted'].append(item['id'])
            persist()
        remaining = client.items()
        assert not ({x['id'] for x in remaining} & set(record['deleted']))
        record.update(complete=True, remaining=len(remaining))
        persist()
        print(json.dumps(dict(deleted=len(record['deleted']),remaining=len(remaining),archive=str(archive))), flush=True)
    finally:
        client.close()


if __name__ == '__main__':
    main()
