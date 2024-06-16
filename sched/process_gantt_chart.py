import matplotlib.pyplot as plt
from process_class import Process

def plot_gantt_chart(process_sequence, preemptive: bool = False, save: bool = False, name: str = 'gantt_chart'):
    fig, ax = plt.subplots(figsize=(10, 2.5))
    
    BAR_HEIGHT: int = 9
    MIN_DISTANCE: float = .8

    first_process = process_sequence[0]

    if preemptive == True:
        for process in process_sequence:
            for (start, end) in process.execution_bursts:
                ax.broken_barh([(start, end)], (MIN_DISTANCE, BAR_HEIGHT), facecolors=('tab:blue'))
                ax.text(start + end / 2, MIN_DISTANCE + BAR_HEIGHT / 2, f'P{process.pid}', 
                        ha='center', va='center', color='white', fontsize=10, fontweight='bold')
    else:
        for process in process_sequence:
            ax.broken_barh([(process.arrival_time + process.waiting_time, process.burst_time)], (MIN_DISTANCE, BAR_HEIGHT), facecolors=('tab:blue'))
            ax.text(process.arrival_time + process.waiting_time + process.burst_time / 2, MIN_DISTANCE + BAR_HEIGHT / 2, f'P{process.pid}', 
                    ha='center', va='center', color='white', fontsize=10, fontweight='bold')
        
    ax.set_ylim(0, MIN_DISTANCE + BAR_HEIGHT + MIN_DISTANCE)
    ax.set_xlim(0, process_sequence[-1].completion_time)

    ax.set_xlabel('Timp [ms]')
    ax.set_ylabel('Procese')

    if preemptive == True:
        ax.set_xticks([0] + [start + end for process in process_sequence for (start, end) in process.execution_bursts])
    else:
        ax.set_xticks([(first_process.arrival_time + first_process.waiting_time)] + [process.completion_time for process in process_sequence])

    if preemptive == True:
        for process in process_sequence:
            for (start, end) in process.execution_bursts:
                # skip last burst from last process
                if process == process_sequence[-1] and (start, end) == process.execution_bursts[-1]:
                    break
                ax.vlines(start + end, ymin=MIN_DISTANCE, ymax=MIN_DISTANCE + BAR_HEIGHT, color='black', linestyles='dashed')
    else:
        for process in process_sequence[:-1]:
            time = process.completion_time
            ax.vlines(time, ymin=MIN_DISTANCE, ymax=MIN_DISTANCE + BAR_HEIGHT, color='black', linestyles='dashed')

    #   hiding y numbers and ticks
    ax.set_yticks([])
    ax.set_yticklabels([])

    plt.title('Planificare Procese')

    fig.tight_layout()

    if save:
        plt.savefig(f'{name}.pdf')

    plt.show()

if __name__ == '__main__':
    plot_gantt_chart()