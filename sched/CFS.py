import random
import sys
from math import floor
from typing import Callable, List

import numpy as np
from sortedcontainers import SortedKeyList
from tabulate import tabulate

from process_class import Process
from get_processes import get_processes

from process_gantt_chart import plot_gantt_chart

import heapq


class CFS:
    def __init__(self):
        self.process_queue = []
        self.current_time = 0

    def add_process(self, process):
        heapq.heappush(self.process_queue, process)

    def execute(self):
        while self.process_queue:
            # Get the process with the smallest vruntime
            process = heapq.heappop(self.process_queue)
            
            # Simulate the process execution
            time_slice = self.calculate_time_slice(process)
            exec_time = min(process.burst_time - process.exec_time, time_slice)

            # Update times
            self.current_time += exec_time
            process.exec_time += exec_time
            process.vruntime += exec_time * self.calc_weight(process.nice)
            
            # Check if process is completed
            if process.exec_time == process.burst_time:
                process.completion_time = self.current_time
                process.turnaround_time = process.completion_time - process.arrival_time
                process.waiting_time = process.turnaround_time - process.burst_time
            else:
                # Reinsert the process into the queue
                heapq.heappush(self.process_queue, process)

    def calculate_time_slice(self, process):
        # Time slice can be adjusted based on system parameters
        return 10  # For simplicity, every process gets a time slice of 10 units

    def calc_weight(self, nice):
        # For simplicity, we'll assume nice values range from -20 to 19
        return 1024 / (1.25 ** nice)

def cfs_schedule(tasks: List[Process], quantum: int) -> None:
    """
    Schedule tasks according to CFS algorithm and set waiting and turnaround times.
    """

    get_vruntime: Callable[[Process], int] = lambda task: task.vruntime
    get_nice: Callable[[Process], int] = lambda task: task.nice

    tasks_sorted = SortedKeyList(key=get_vruntime)
    tasks_sorted.add(tasks[0])
    end = 1
    timer = tasks[0].arrival_time
    min_vruntime = 0

    while (num := len(tasks_sorted)) > 0:
        # Add tasks that have arrived after previous iteration
        for task in tasks[end:]:
            if task.arrival_time <= timer:
                task.waiting_time = timer - task.arrival_time
                task.turnaround_time = task.waiting_time
                task.vruntime = min_vruntime
                tasks_sorted.add(task)
                num += 1
                end += 1

        timeslice = quantum / num
        min_task = tasks_sorted[0]

        # Time remaining for smallest task
        t_rem = min_task.burst_time - min_task.exec_time


        # MAY NEED TO MODIFY, WE ARE USING 'CPU BURST TIME' INSTEAD OF 'TIMESLICE'
        # MAY NEED TO MODIFY, WE ARE USING 'CPU BURST TIME' INSTEAD OF 'TIMESLICE'
        # MAY NEED TO MODIFY, WE ARE USING 'CPU BURST TIME' INSTEAD OF 'TIMESLICE'
        # MAY NEED TO MODIFY, WE ARE USING 'CPU BURST TIME' INSTEAD OF 'TIMESLICE'
        # MAY NEED TO MODIFY, WE ARE USING 'CPU BURST TIME' INSTEAD OF 'TIMESLICE'
        # MAY NEED TO MODIFY, WE ARE USING 'CPU BURST TIME' INSTEAD OF 'TIMESLICE'
        # MAY NEED TO MODIFY, WE ARE USING 'CPU BURST TIME' INSTEAD OF 'TIMESLICE'
        # MAY NEED TO MODIFY, WE ARE USING 'CPU BURST TIME' INSTEAD OF 'TIMESLICE'
        # MAY NEED TO MODIFY, WE ARE USING 'CPU BURST TIME' INSTEAD OF 'TIMESLICE'
        # MAY NEED TO MODIFY, WE ARE USING 'CPU BURST TIME' INSTEAD OF 'TIMESLICE'
        # MAY NEED TO MODIFY, WE ARE USING 'CPU BURST TIME' INSTEAD OF 'TIMESLICE'

        # Time of execution of smallest task
        time = min([timeslice, t_rem])              ############################# MAY NEED TO MODIFY
        min_vruntime = get_vruntime(min_task)
        min_nice = get_nice(min_task)

        # display_tasks(tasks_sorted)
        # print(f'Executing task {min_task["pid"]} for {time} seconds\n')

        # Execute process
        vruntime = min_vruntime + time * min_nice
        min_task.exec_time += time
        min_task.turnaround_time += time
        timer += time

        # Increment waiting and turnaround time of all other processes
        for i in range(1, num):
            task = tasks_sorted[i]
            task.waiting_time += time
            task.turnaround_time += time

        # Remove from sorted list and update vruntime
        task = tasks_sorted.pop(0)
        task.vruntime = vruntime

        # Insert only if execution time is left
        if min_task.exec_time < min_task.burst_time:
            tasks_sorted.add(task)

    for task in tasks:
        task.completion_time = task.arrival_time + task.turnaround_time

def display_tasks(tasks: List[dict]):
    """
    Print all tasks' information in a table.
    """

    headers = [
        "ID",
        "Arrival Time",
        "Burst Time",
        "Nice",
        "Waiting Time",
        "Turnaround Time",
    ]
    tasks_mat = []

    for task in tasks:
        tasks_mat.append(
            [
                task.pid,
                f"{task.arrival_time / 1000}",
                f"{task.burst_time / 1000}",
                task.nice,
                f"{task.waiting_time / 1000}",
                f"{task.turnaround_time / 1000}",
            ]
        )
    print(
        "\n"
        + tabulate(tasks_mat, headers=headers, tablefmt="fancy_grid", floatfmt=".3f")
    )
    # print('\n' + tabulate(tasks, headers='keys', tablefmt='fancy_grid'))


def find_avg_time(tasks: List[Process]):
    """
    Find average waiting and turnaround time.
    """

    waiting_times = []
    total_wt = 0
    total_tat = 0
    num = len(tasks)

    for task in tasks:
        waiting_times.append(task.waiting_time)
        total_wt += task.waiting_time
        total_tat += task.turnaround_time

    print(f"\nAverage waiting time: {total_wt: .3f} seconds")
    print(f"Average turnaround time: {total_tat: .3f} seconds")
    print(
        "Standard deviation in waiting time: "
        f"{np.std(waiting_times): .3f} seconds"
    )


def get_tasks(N: int, MAX_ARRIVAL_TIME: int, MAX_BURST_TIME: int, MAX_NICE_VALUE: int) -> List[Process]:
     
    # print('Enter ID, arrival time, burst time, nice value of processes:')
    # print('(Times should be in milliseconds)')

    tasks = []

    for _ in range(N):
        # pid, at, bt, nice = tuple(int(x) for x in input().split())
        pid, at, bt, nice = (
            random.randint(1, N * N),
            random.randint(0, MAX_ARRIVAL_TIME),
            random.randint(0, MAX_BURST_TIME),
            random.randint(1, MAX_NICE_VALUE),
        )
        tasks.append(
            Process(
                pid=pid,
                arrival_time=at,
                burst_time=bt,
                nice=nice,
                waiting_time=0,
                turnaround_time=0,
                exec_time=0,
                vruntime=0,
            )
        )

    return tasks

if __name__ == "__main__":
    MIN_VERSION = (3, 8)
    if not sys.version_info >= MIN_VERSION:
        raise EnvironmentError(
            "Python version too low, required at least "
            f'{".".join(str(n) for n in MIN_VERSION)}'
        )

    # N = int(input("Enter number of tasks: "))

    QUANTUM = 200  # Time quantum in ms
    # MAX_ARRIVAL_TIME = 20_000
    # MAX_BURST_TIME = 50_000
    # MAX_NICE_VALUE = 10
    # TASKS = get_tasks(N, MAX_ARRIVAL_TIME, MAX_BURST_TIME, MAX_NICE_VALUE)

    TASKS = [
        Process(pid=1, burst_time=6, arrival_time=2),
        Process(pid=2, burst_time=8, arrival_time=5),
        Process(pid=3, burst_time=7, arrival_time=1),
        Process(pid=4, burst_time=3, arrival_time=0),
    ]

    # processes = get_processes()

    # original_time = processes[0][1]

    # TASKS = [Process(pid=i, burst_time=burst_time, arrival_time=arrival_time - original_time) for i, (burst_time, arrival_time) in enumerate(processes)]

    # Sort tasks by arrival time
    # TASKS_SORTED = SortedKeyList(TASKS, key=lambda task: task.arrival_time)


    TASKS = sorted(TASKS, key=lambda task: task.arrival_time)

    cfs = CFS()

    for task in TASKS:
        cfs.add_process(task)

    cfs.execute()

    avg_turnaround_time = sum(process.turnaround_time for process in TASKS) / len(TASKS)
    avg_waiting_time = sum(process.waiting_time for process in TASKS) / len(TASKS)
        

    print(f"\nAverage Turnaround Time: {avg_turnaround_time}")
    print(f"Average Waiting Time: {avg_waiting_time}")

    # Schedule tasks according to CFS algorithm and print average times
    # cfs_schedule(TASKS_SORTED, QUANTUM)
    # print("\n**************** CFS SCHEDULING ****************")
    # display_tasks(TASKS)
    # find_avg_time(TASKS)

    # for task in TASKS:
    #     print(task)

    # plot_gantt_chart(TASKS, save=False)