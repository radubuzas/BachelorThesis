from typing import List, Tuple

class Process:
    def __init__(self, pid, burst_time, arrival_time, completion_time=0, turnaround_time=0, waiting_time=0, vruntime=0, nice=0):
        self.pid = pid
        self.burst_time = burst_time
        self.arrival_time = arrival_time
        self.completion_time = completion_time
        self.turnaround_time = turnaround_time
        self.waiting_time = waiting_time
        self.execution_bursts: List[Tuple[int, int]] = []
        self.vruntime = vruntime
        self.nice = nice

        self.predicted_burst = burst_time

    def __lt__(self, other):
        if self.vruntime == 0:
            return self.predicted_burst < other.predicted_burst
        
        return self.vruntime < other.vruntime
    
    
    def __str__(self):
        return f"Process {self.pid}:\nBurst Time: {self.burst_time}\nArrival Time: {self.arrival_time}\nCompletion Time: {self.completion_time}\nTurnaround Time: {self.turnaround_time}\nWaiting Time: {self.waiting_time}\nBurts: {self.execution_bursts}\n"