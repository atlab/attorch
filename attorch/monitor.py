from collections import defaultdict

import gpustat
import threading
import numpy as np
import subprocess

#TODO: This is buggy. gpustat fails with "Command 'ps -o pid,user:16,comm -p 26714' returned non-zero exit status 1"
# when the memory increases on the GPU
def memusage(stats, device=0, interval=0.1, stop_event = None):
    while (not stop_event.is_set()):
        try:
            gpu_stats = gpustat.GPUStatCollection.new_query()
            item = gpu_stats.jsonify()["gpus"][device]
            stats['time'].append(gpu_stats.query_time)
            stats['used'].append(item["memory.used"])
            stats['total'].append(item["memory.total"])
            stop_event.wait(interval)
        except subprocess.CalledProcessError as e:
            print(e)

class MemoryUsage:

    def __init__(self, device=0, interval=0.1):
        self.devive = device
        self.interval = interval
        self.thread = None
        self.stats = defaultdict(list)
        self._stop_event = None

    def clear(self):
        self.stats = defaultdict(list)

    def __enter__(self):
        self.clear()
        self._stop_event= threading.Event()
        self.thread = threading.Thread(target=memusage, args=(self.stats, self.devive, self.interval, self._stop_event))
        self.thread.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        self.stats['used'] = np.array(list(map(float, self.stats['used'])))
        self.stats['total'] = np.array(list(map(float, self.stats['total'])))


    def plot(self):
        import matplotlib.pyplot as plt
        import seaborn as sns

