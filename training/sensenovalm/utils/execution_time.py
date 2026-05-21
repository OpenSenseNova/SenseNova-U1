# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
import time


class _TimeContext:
    """Context manager for measuring execution time of a block of code."""

    def __init__(self, loader_timer, timer_name):
        self.loader_timer = loader_timer
        self.timer_name = timer_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, *args, **kwargs):
        elapsed_time = time.time() - self.start_time
        self.loader_timer._update_timer(self.timer_name, elapsed_time)


class ExecutionTimeCollector:
    """Collect execution time of a block of code."""

    def __init__(self):
        self.timers = {}

    def collect_execute_time(self, timer_name):
        return _TimeContext(self, timer_name)

    def _update_timer(self, timer_name, elapsed_time):
        if timer_name not in self.timers:
            self.timers[timer_name] = 0.0
        self.timers[timer_name] += elapsed_time

    def __getattr__(self, timer_name):
        if timer_name in self.timers:
            return self.timers[timer_name]
        raise AttributeError(f"'LoaderTimer' object has no attribute '{timer_name}'")


execution_time_collecter = ExecutionTimeCollector()

if __name__ == "__main__":
    etc = execution_time_collecter

    with etc.collect_execute_time("test"):
        time.sleep(0.5)

    print(etc.test)
