from get_processes import *
from process_class import *
from process_gantt_chart import *

import queue

def rr_schedule(processes: list[Process], quantum: int=4):
    """
    Schedule tasks according to preemptive Round Robin algorithm and set waiting and
    turnaround times.
    """

    n: int = len(processes)

    processes.sort(key=lambda x: x.arrival_time)

    processes.append(Process(-1, 0, 1e9))    # Append a dummy process to avoid index out of range error

    ready_queue = queue.Queue()

    last_process_index: int = 0

    for process in processes:
        if process.arrival_time != 0:
            break
        ready_queue.put(process)
        last_process_index += 1

    current_time = 0

    while last_process_index <= n:
        while not ready_queue.empty():
            current_process = ready_queue.get()

            if current_process.burst_time > quantum:
                current_process.execution_bursts.append((current_time, quantum))
                current_time += quantum
                current_process.burst_time -= quantum

                # if process arrives while executing, add them to the ready queue
                if current_time > processes[last_process_index].arrival_time:
                    while last_process_index != n and processes[last_process_index].arrival_time <= current_time:
                        ready_queue.put(processes[last_process_index])
                        last_process_index += 1

                ready_queue.put(current_process)
                while last_process_index != n and processes[last_process_index].arrival_time <= current_time:
                    ready_queue.put(processes[last_process_index])
                    last_process_index += 1

            else:
                current_process.execution_bursts.append((current_time, current_process.burst_time))
                current_time += current_process.burst_time
                current_process.burst_time = 0
                current_process.turnaround_time = current_time - current_process.arrival_time
                current_process.waiting_time = current_process.turnaround_time - current_process.predicted_burst
                current_process.completion_time = current_time

        if last_process_index < n:
            if current_time > processes[last_process_index].arrival_time:
                while last_process_index != n and processes[last_process_index].arrival_time <= current_time:
                    ready_queue.put(processes[last_process_index])
                    last_process_index += 1
            else:
                ready_queue.put(processes[last_process_index])
                current_time = processes[last_process_index].arrival_time
                last_process_index += 1
        else:
            break

    avg_turnaround_time = sum([process.turnaround_time for process in processes]) / n
    avg_waiting_time = sum([process.waiting_time for process in processes]) / n
    
    print(f"\nAverage Turnaround Time: {avg_turnaround_time}")
    print(f"Average Waiting Time: {avg_waiting_time}")

    return avg_turnaround_time, avg_waiting_time

def main():
    # processes = get_processes()

    # original_time = processes[0][1]

    # print('Creating processes...')

    # processes = [Process(i, burst_time, arrival_time - original_time) for i, (burst_time, arrival_time) in enumerate(processes)]

    # processes = [
    #     Process(pid=1, burst_time=6, arrival_time=2),
    #     Process(pid=2, burst_time=8, arrival_time=5),
    #     Process(pid=3, burst_time=7, arrival_time=1),
    #     Process(pid=4, burst_time=3, arrival_time=0),
    # ]

    
    rr_schedule(processes, quantum=quantum)

    # plot_gantt_chart(processes[:-1], preemptive=True, name='rr_gantt_chart', save=True)

if __name__ == '__main__':
    main()

    