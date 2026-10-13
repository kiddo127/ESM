#!/usr/bin/env python3
"""Dispatch the MoE route-fix matrix (modes x tasks) over a pool of GPUs.

Runs run_moe_routefix_task.sh for each (mode, task) pair, pinned to a free GPU
from the pool. Longest jobs (coding, safety) are submitted first.

Usage (on the GPU node): python dispatch_moe_routefix.py 0,1,2,3,4,5,6 [modes] [tasks]
"""
import os
import queue
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(os.environ.get('PROJECT_ROOT', Path(__file__).resolve().parent))
ALL_MODES = ['base', 'oracle', 'fix_agg']
ALL_TASKS = ['math', 'instruction', 'multilingual', 'coding', 'safety']
COST = {'coding': 5, 'safety': 4, 'math': 3, 'multilingual': 3, 'instruction': 2}


def main():
    gpus = [int(g) for g in (sys.argv[1] if len(sys.argv) > 1 else '0').split(',')]
    modes = sys.argv[2].split(',') if len(sys.argv) > 2 else ALL_MODES
    tasks = sys.argv[3].split(',') if len(sys.argv) > 3 else ALL_TASKS

    jobs = [(m, t) for m in modes for t in tasks]
    # Short jobs first -> early signal; heavy jobs (coding, safety) drain last.
    jobs.sort(key=lambda j: COST.get(j[1], 1))

    gpu_q = queue.Queue()
    for g in gpus:
        gpu_q.put(g)
    lock = threading.Lock()
    logs = ROOT / 'bench' / 'logs'
    logs.mkdir(parents=True, exist_ok=True)

    def run(job):
        mode, task = job
        gpu = gpu_q.get()
        log = logs / f'routefix_{mode}_{task}.log'
        env = dict(os.environ, MODE=mode, TASK=task, GPU_ID=str(gpu))
        try:
            with open(log, 'w') as f:
                f.write(f'# MODE={mode} TASK={task} GPU={gpu}\n')
                f.flush()
                rc = subprocess.run(['bash', str(ROOT / 'run_moe_routefix_task.sh')],
                                    cwd=str(ROOT), env=env, stdout=f,
                                    stderr=subprocess.STDOUT).returncode
        finally:
            gpu_q.put(gpu)
        with lock:
            print(f"  [{'ok' if rc == 0 else f'FAIL({rc})'}] {mode}/{task} (gpu {gpu})", flush=True)
        return job, rc

    print(f"=== dispatch: {len(jobs)} jobs on GPUs {gpus} ===", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        results = list(ex.map(run, jobs))
    bad = [j for j, rc in results if rc != 0]
    print(f"=== done: {len(results) - len(bad)} ok, {len(bad)} failed {bad} ===", flush=True)


if __name__ == '__main__':
    main()
