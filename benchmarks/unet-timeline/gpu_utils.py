from collections import deque
from contextlib import contextmanager
import multiprocessing as mp
import subprocess
from typing import Dict, Generator, List, cast


def collect_gpu_utils(device_ids: List[int]) -> List[int]:
    """Collects GPU% via nvidia-smi. It expects that nvidia-smi v418.43 is
    executable in the running system.
    """
    unique_device_ids = deque(sorted(set(device_ids)))

    # Execute nvidia-smi.
    i_arg = ','.join(str(i) for i in unique_device_ids)
    cmd = ['nvidia-smi', '-q', '-i', i_arg, '-d', 'utilization']
    p = subprocess.run(cmd, stdout=subprocess.PIPE)

    # Parse GPU utilizations.
    gpu_utils: Dict[int, int] = {}
    for line in p.stdout.decode().split('\n'):
        if 'Gpu' in line:
            for word in line.split():
                if word.isdigit():
                    i = unique_device_ids.popleft()
                    gpu_utils[i] = int(word)

    return [gpu_utils[i] for i in device_ids]


def _worker(device_ids: List[int],
            interval: float,
            conn: mp.connection.Connection,
            ) -> None:
    gpu_timeline: List[List[int]] = []
    conn.send(None)

    while not conn.poll(timeout=interval):
        gpu_utils = collect_gpu_utils(device_ids)
        gpu_timeline.append(gpu_utils)

    conn.send(gpu_timeline)


@contextmanager
def track_gpu_utils(device_ids: List[int],
                    interval: float = 0.05,
                    ) -> Generator[List[float], None, None]:
    # Spawn a worker.
    ctx = mp.get_context('spawn')
    ctx = cast(mp.context.DefaultContext, ctx)
    conn, conn_worker = ctx.Pipe(duplex=True)
    p = ctx.Process(target=_worker, args=(device_ids, interval, conn_worker))
    p.start()
    conn.recv()

    # GPU% will be filled to this.
    gpu_utils: List[float] = []
    yield gpu_utils

    # Stop the worker and receive the timeline.
    conn.send(None)
    gpu_timeline = conn.recv()
    p.join()

    # Fill the GPU%.
    if gpu_timeline:
        gpu_utils.extend(sum(t)/len(t)/100 for t in zip(*gpu_timeline))
    else:
        gpu_utils.extend(0.0 for _ in device_ids)
