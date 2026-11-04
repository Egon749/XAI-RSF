"""Summarize completed task affinity; do not change any CPU binding."""
from collections import Counter
from pathlib import Path
import socket

from .common import load, run_path, write


def cpu_list(path):
    result = set()
    for part in Path(path).read_text().strip().split(','):
        ends = list(map(int, part.split('-')))
        result.update(range(ends[0], ends[-1] + 1))
    return result


def main():
    if socket.gethostname() != 'oliverdesktop':
        raise RuntimeError('Run on oliver to read its own kernel core-type lists')
    pc = cpu_list('/sys/bus/event_source/devices/cpu_core/cpus')
    ec = cpu_list('/sys/bus/event_source/devices/cpu_atom/cpus')
    assert pc.isdisjoint(ec)
    root = run_path()
    stages = []
    for stage in ('pilot', 'formal', 'ensemble'):
        gate = load(root / f'audit/gate_{stage}.json')
        counts = Counter()
        for task in load(gate['result'])['completed']:
            assert task['compute_completed'] and task['publication_completed']
            assert task['kind'] == 'cpu' and len(task['cpus']) == 1
            cpu = task['cpus'][0]
            if task['host'] == 'lemon':
                label = 'lemon_CPU'
            else:
                assert task['host'] == 'oliverdesktop' and cpu in pc | ec
                label = 'oliver_P' if cpu in pc else 'oliver_E'
            counts[label] += 1
        stages.append(dict(stage=stage, job=gate['job'], task_counts=dict(counts)))
    value = dict(source='Slurm heteropool per-task affinity; oliver kernel cpu_core/cpu_atom CPU lists. Counts are completed tasks, not simultaneous active core counts.', stages=stages)
    path = root / 'reports/CPU_PLACEMENT.json'
    if path.exists():
        assert load(path) == value, 'Existing placement record differs'
    else:
        write(path, value)


if __name__ == '__main__':
    main()
