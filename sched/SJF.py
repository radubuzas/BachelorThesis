from process_gantt_chart import plot_gantt_chart
from get_processes import get_processes
from queue import PriorityQueue 
from copy import deepcopy

from RR import rr_schedule

import types, typing

import numpy as np
import random

from process_class import Process

times: list[tuple[float, float]] = []

def calculate_fcfs(processes) -> list[Process]:
    n = len(processes)

    processes.sort(key=lambda x: x.arrival_time)
    processes.append(Process(-1, 0, 1e9))    # Append a dummy process to avoid index out of range error
    last_process_index = 0

    completed = 0
    current_time = 0

    dead_time: float = 0

    process_sequence = []

    for process in processes:
        if process.pid == -1:
            break
        current_time += process.burst_time

        if current_time < process.arrival_time:
            dead_time += process.arrival_time - current_time
            current_time = process.arrival_time

        process.completion_time = current_time
        process.turnaround_time = process.completion_time - process.arrival_time
        process.waiting_time = process.turnaround_time - process.burst_time

        process_sequence.append(process)

        completed += 1

    avg_turnaround_time = sum(process.turnaround_time for process in processes) / n
    avg_waiting_time = sum(process.waiting_time for process in processes) / n
    
    print(f"\nAverage Turnaround Time: {avg_turnaround_time}")
    print(f"Average Waiting Time: {avg_waiting_time}")
    print(f"Idle Time: {dead_time}")

    return avg_turnaround_time, process_sequence

class random_queue:
    def __init__(self, processes: list[Process] = []):
        self.processes = processes

    def put(self, process: Process):
        self.processes.append(process)

    def get(self):
        return self.processes.pop(random.randint(0, len(self.processes) - 1))

    def empty(self):
        return len(self.processes) == 0

    def __len__(self):
        return len(self.processes)

    def __str__(self):
        return str(self.processes)

    def __repr__(self):
        return str(self.processes)

    def __iter__(self):
        return iter(self.processes)

    def __next__(self):
        if self.last_process_index < len(self.processes):
            process = self.processes[self.last_process_index]
            self.last_process_index += 1
            return process
        else:
            raise StopIteration

    def __getitem__(self, key):
        return self.processes[key]

    def __setitem__(self, key, value):
        self.processes[key] = value

    def __delitem__(self, key):
        del self.processes[key]

    def __contains__(self, item):
        return item in self.processes

    def __add__(self, other):
        return self.processes + other

    def __iadd__(self, other):
        self.processes += other
        return self

    def __eq__(self, other):
        return self.processes == other

    def __ne__(self, other):
        return self.processes != other

    def __lt__(self, other):
        return self.processes < other

    def __le__(self, other):
        return self.processes <= other

    def __gt__(self, other):
        return self.processes > other

    def __ge__(self, other):
        return self.processes >= other

    def __hash__(self):
        return hash(self.processes)

def calculate_sjf(processes: list[Process], random: bool = False) -> tuple[int, list[Process]]:

    n = len(processes)

    processes.sort(key=lambda x: (x.arrival_time))    # Sort processes by arrival time and burst time
    processes.append(Process(-1, 0, 1e9))    # Append a dummy process to avoid index out of range error
    last_process_index = 0

    completed = 0
    current_time = 0

    dead_time: float = 0

    process_sequence = []

    if random:
        ready_queue = random_queue([])
    else:
        ready_queue = PriorityQueue()

    for process in processes:
        if process.arrival_time != 0:
            break
        ready_queue.put(process)
        last_process_index += 1

    VERY_SMALL_PENALITY = 0.1
    PENALITY = 0.2
    CONTEXT_SWITCH = 0.5

    misses: int = 0
    wasted_time: float = 0


    #permutare vector procese

    # cateva permutari pentru a verifica performanta

    while completed != n:   # While all processes are not completed
        while not ready_queue.empty() and current_time < processes[last_process_index].arrival_time:    # No need to add new processes to ready queue 
            current_process = ready_queue.get()
            # if prediction_mode:
            #     if current_process.burst_time > current_process.predicted_burst:
            #         if current_process.burst_time - current_process.predicted_burst > CONTEXT_SWITCH:
            #             extra += PENALITY
            #         else:
            #             extra += VERY_SMALL_PENALITY
            #     else:
            #         if current_process.predicted_burst - current_process.burst_time <= CONTEXT_SWITCH:
            #             extra += current_process.predicted_burst - current_process.burst_time
            #         else:
            #             extra += PENALITY



            current_time += current_process.burst_time

            current_process.completion_time = current_time
            current_process.turnaround_time = current_process.completion_time - current_process.arrival_time
            current_process.waiting_time = current_process.turnaround_time - current_process.burst_time

            process_sequence.append(current_process)

            completed += 1

            if completed == n:
                break

        if completed == n:
            break
        
        if ready_queue.empty(): # If no process is available in ready queue, move the next porcesses to the ready queue
            if processes[last_process_index].arrival_time > current_time:
                dead_time += processes[last_process_index].arrival_time - current_time
                current_time = processes[last_process_index].arrival_time
            while last_process_index < n and processes[last_process_index].arrival_time <= current_time:    # Add all processes that have arrived at the current time
                ready_queue.put(processes[last_process_index])
                last_process_index += 1
            continue
        else:
            while last_process_index < n and processes[last_process_index].arrival_time <= current_time:
                ready_queue.put(processes[last_process_index])
                last_process_index += 1

    
    avg_turnaround_time = sum(process.turnaround_time for process in processes) / n
    avg_waiting_time = sum(process.waiting_time for process in processes) / n
    
    
    print(f"\nAverage Turnaround Time: {avg_turnaround_time}")
    print(f"Average Waiting Time: {avg_waiting_time}")
    print(f"Idle Time: {dead_time}")

    times.append((avg_turnaround_time, avg_waiting_time))

    return avg_turnaround_time, process_sequence

def main():
    # Example usage
    # processes = [
    #     Process(pid=1, burst_time=6, arrival_time=2),
    #     Process(pid=2, burst_time=8, arrival_time=5),
    #     Process(pid=3, burst_time=7, arrival_time=1),
    #     Process(pid=4, burst_time=3, arrival_time=0),
    # ]

    ########################################################################################

    fake_times, processes = get_processes(get_samples=True)

    original_time = processes[0][1]

    print('Creating processes...')

    processes = [Process(i, burst_time, arrival_time - original_time) for i, (burst_time, arrival_time) in enumerate(processes)]

    print('Processes created!')

    sol = []

    print('Calculating SJF...')
    x, process_sequence_sjf = calculate_sjf(deepcopy(processes))
    sol.append(x)

    print('\n\nCalculating SJF with predictions!')

    processes_ml = deepcopy(processes)


    # ################################################################################################################################################################################################################################################
    # ############################################################
    # #################### START OF 30_000 #######################
    # ############################################################
    # ############################################################

    # predictions = np.load('../predictions_30000_linear_loo_0.666.npy')
    # for i, process in enumerate(processes_ml):
    #     process.predicted_burst = predictions[i] 
    # print('\n\nLoaded Linear loo - 30_000!')
    # x, _ = calculate_sjf(deepcopy(processes_ml))
    # sol.append(x)

    # ############################################################
    # ############################################################

    # predictions = np.load('../predictions_30000_knn_loo_0.509.npy')
    # for i, process in enumerate(processes_ml):
    #     process.predicted_burst = predictions[i] 
    # print('\n\nLoaded KNN loo - 30_000!')
    # x, _ = calculate_sjf(deepcopy(processes_ml))
    # sol.append(x)

    # ############################################################
    # ############################################################

    # predictions = np.load('../predictions_30000_rt_loo_0.751.npy')
    # for i, process in enumerate(processes_ml):
    #     process.predicted_burst = predictions[i] 
    # print('\n\nLoaded RT loo - 30_000!')
    # x, _ = calculate_sjf(deepcopy(processes_ml))
    # sol.append(x)

    # ############################################################
    # ############################################################

    # predictions = np.load('../predictions_30000_rf_loo_0.764.npy')
    # for i, process in enumerate(processes_ml):
    #     process.predicted_burst = predictions[i] 
    # print('\n\nLoaded RF loo - 30_000!')
    # x, _ = calculate_sjf(deepcopy(processes_ml))
    # sol.append(x)

    # ############################################################
    # ############################################################

    # predictions = np.load('../predictions_30000_xgboost_loo_0.768.npy')
    # for i, process in enumerate(processes_ml):
    #     process.predicted_burst = predictions[i] 
    # print('\n\nLoaded XGBOOST loo - 30_000!')
    # x, _ = calculate_sjf(deepcopy(processes_ml))
    # sol.append(x)

    # ############################################################
    # ############################################################

    # print('\n\nLoaded XGBOOST loo - 30_000!')
    # x, _ = rr_schedule(deepcopy(processes))
    # sol.append(x)

    # ############################################################

    print('\n\nCalculating SJF with sampled times...')
    processes_sample_times = deepcopy(processes)

    for i, process in enumerate(processes_sample_times):
        process.predicted_burst = fake_times[i]

    x, _ = calculate_sjf(deepcopy(processes_sample_times))
    sol.append(x)

    # ############################################################
    # ############################################################

    # for i in range(2):
    #     print('\n\nCalculating SJF with random order...')
    #     x, _ = calculate_sjf(deepcopy(processes), random=True)
    #     sol.append(x)
    
    # ############################################################

    # ############################  PLOT  #############################

    # import matplotlib.pyplot as plt
    # import seaborn as sns

    # sns.set_theme(style='whitegrid')

    # plt.plot(sol)

    # plt.yscale('log')

    # plt.ylabel('Timpul mediu de execuție [s]')
    # plt.xlabel('Tipuri algoritm')

    # plt.title('Comparație algoritmi')

    # plt.xticks(np.arange(10), ['SJF', 'Linear', 'KNN', 'RT', 'RF', 'XGBOOST', 'RR', 'SJF eșantionate', 'SJF random', 'SJF random'])

    # plt.xticks(rotation=-45)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('comparatie.pdf')
    # plt.show()

    # #######################END PLOT#############################

    # ############################################################
    # ############################################################
    # #################### END OF 30_000 #########################
    # ############################################################
    # ################################################################################################################################################################################################################################################


    # ################################################################################################################################################################################################################################################
    # ############################################################
    # ###################### START OF 10% ########################
    # ############################################################
    # ############################################################

    # predictions = np.load('../predictions_10_linear_loo_0.819.npy')
    # for i, process in enumerate(processes_ml):
    #     process.predicted_burst = predictions[i] 
    # print('\n\nLoaded Linear loo - 10%!')
    # x, _ = calculate_sjf(deepcopy(processes_ml))
    # sol.append(x)

    # ############################################################
    # ############################################################

    # predictions = np.load('../predictions_knn_target_0.782.npy')
    # for i, process in enumerate(processes_ml):
    #     process.predicted_burst = predictions[i]
    # print('\n\nLoaded KNN target - 10%!')
    # x, calculate_sjf(deepcopy(processes_ml))
    # sol.append(x)

    # ############################################################
    # ############################################################

    # predictions = np.load('../predictions_10_rt_loo_0.856.npy')
    # for i, process in enumerate(processes_ml):
    #     process.predicted_burst = predictions[i] 
    # print('\n\nLoaded RT loo - 10%!')
    # x, _ = calculate_sjf(deepcopy(processes_ml))
    # sol.append(x)

    # ############################################################
    # ############################################################

    # predictions = np.load('../predictions_30000_rf_loo_0.764.npy')
    # for i, process in enumerate(processes_ml):
    #     process.predicted_burst = predictions[i] 
    # print('\n\nLoaded RF loo - 10%!')
    # x, _ = calculate_sjf(deepcopy(processes_ml))
    # sol.append(x)

    # ############################################################
    # ############################################################

    # predictions = np.load('../predictions_10_xgboost_loo_.854.npy')
    # for i, process in enumerate(processes_ml):
    #     process.predicted_burst = predictions[i] 
    # print('\n\nLoaded XGBOOST loo - 30_000!')
    # x, _ = calculate_sjf(deepcopy(processes_ml))
    # sol.append(x)

    # ############################################################
    # ############################  PLOT  #############################

    # import matplotlib.pyplot as plt
    # import seaborn as sns

    # sns.set_theme(style='whitegrid')

    # plt.plot(sol)

    # plt.yscale('log')

    # plt.ylabel('Timpul mediu de execuție [s]')
    # plt.xlabel('Tipuri algoritm')

    # plt.title('Comparație algoritmi')

    # plt.xticks(np.arange(6), ['SJF', 'Linear', 'KNN', 'RT', 'RF', 'XGBOOST'])
    
    # plt.xticks(rotation=-45)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('comparatie.pdf')
    # plt.show()

    # #######################END PLOT#############################

    # ############################################################
    # ############################################################
    # ###################### END OF 10% ##########################
    # ############################################################
    # ################################################################################################################################################################################################################################################


    # predictions = np.load('../predictions_xgboost_loo_89.npy')
    # for i, process in enumerate(processes_ml):
    #     process.predicted_burst = predictions[i] 
    # print('\n\nLoaded XGBOOST loo - 10%!')
    # x, _ = calculate_sjf(deepcopy(processes_ml))
    # sol.append(x)

    ############################################################

    # predictions = np.load('../predictions_xgboost_loo_88.npy')
    # for i, process in enumerate(processes_ml):
    #     process.predicted_burst = predictions[i]
    # print('\n\nLoaded XGBOOST loo - 10%!')
    # x, _ = calculate_sjf(deepcopy(processes_ml))
    # sol.append(x)

    # predictions = np.load('../predictions_linear_leave_one_out_0.82.npy')
    # for i, process in enumerate(processes_ml):
    #     process.predicted_burst = predictions[i]
    # print('\n\nLoaded Linear loo - 10%!')
    # x, _ = calculate_sjf(deepcopy(processes_ml))
    # sol.append(x)


    # predictions = np.load('../predictions_rt_target_93.npy')
    # for i, process in enumerate(processes_ml):
    #     process.predicted_burst = predictions[i] 
    # print('\n\nLoaded RT target - 10%!')
    # x, _ = calculate_sjf(deepcopy(processes_ml))
    # sol.append(x)

    # predictions = np.load('../predictions_rt_leave_one_out_0.751.npy')
    # for i, process in enumerate(processes_ml):
    #     process.predicted_burst = predictions[i]
    # print('\n\nLoaded RT target - 30_000!')

    # x, _ = calculate_sjf(deepcopy(processes_ml))
    # sol.append(x)


    # predictions = np.load('../predictions_rt_leave_one_out_0.751.npy')
    # for i, process in enumerate(processes_ml):
    #     process.predicted_burst = predictions[i]
    # print('\n\nLoaded RT loo - 30_000!')

    # x, _ = calculate_sjf(deepcopy(processes_ml))
    # sol.append(x)

    # predictions = np.load('../predictions_xgb_leave_one_out_0.747.npy')
    # for i, process in enumerate(processes_ml):
    #     process.predicted_burst = predictions[i]
    # print('\n\nLoaded XGBOOST loo - 30_000!')


    # print('\n\nCalculating SJF with sampled times...')
    # processes_sample_times = deepcopy(processes)

    # for i, process in enumerate(processes_sample_times):
    #     process.predicted_burst = fake_times[i]

    # x, _ = calculate_sjf(deepcopy(processes_sample_times))
    # sol.append(x)



    # for i in range(2):
    #     print('\n\nCalculating SJF with random order...')
    #     x, _ = calculate_sjf(deepcopy(processes), random=True)
    #     sol.append(x)


    # print('\n\nCalculating FCFS...')
    # x, process_sequence_fcfs =  calculate_fcfs(deepcopy(processes))

    # sol.append(x)


    ############################  PLOT  #############################

    # import matplotlib.pyplot as plt
    # import seaborn as sns

    # sns.set_theme(style='whitegrid')

    # plt.plot(sol)

    # plt.yscale('log')

    # plt.ylabel('Timpul mediu de execuție [s]')
    # plt.xlabel('Tipuri algoritm')

    # plt.title('Comparație algoritmi')

    # # plt.xticks(np.arange(12), ['SJF', 'SJF cu predicții', 'SJF random', 'SJF random','SJF random','SJF random','SJF random','SJF random','SJF random','SJF random','SJF random','FCFS'])
    # plt.xticks(rotation=45)
    # plt.grid(True)
    # plt.show()

    # plt.tight_layout()

    #######################END PLOT#############################


    # plt.savefig('comparatie.pdf')

    # _, process_sequence_sjf = calculate_sjf(processes)

    # plot_gantt_chart(process_sequence_sjf, 'sjf_gantt_chart', save=False)
    # plot_gantt_chart(process_sequence_fcfs, 'fcfs_gantt_chart')

    # np.savetxt('sjf.csv', [[process.pid, process.burst_time, process.arrival_time, process.completion_time, process.turnaround_time, process.waiting_time] for process in process_sequence], delimiter=',', fmt=('%d', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f'))

if __name__ == '__main__':
    main()
    