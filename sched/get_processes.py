import pandas as pd
import sqlite3

from typing import List, Tuple, Any
from process_class import Process


def get_processes(get_samples=False) -> Tuple[List[int], List[Tuple[int, int ,int]]]:
    print('Awaiting connection...')
    cnx = sqlite3.connect('../anon_jobs.db3')

    print('Connected to database!')

    print('Reading data from database...')
    df = pd.read_sql_query("SELECT * FROM 'Jobs';", cnx)
    print('Done reading data from database!\n\n')

    df = df.sort_values(by=['SubmitTime'])

    if get_samples == True:
        fake_times = df.sample(frac=1,weights=df['UsedCPUTime'].map(df['UsedCPUTime'].value_counts()), replace=True, random_state=1233123)['UsedCPUTime'].values.tolist()

    processes = df[['UsedCPUTime', 'SubmitTime']].values.tolist()


    if get_samples == True:
        return fake_times, processes

    return processes
