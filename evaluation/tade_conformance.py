# Description: This file contains the implementation of the TADE conformance checking algorithm.
# Code adapted from: https://github.com/Skarvir/TADE


import pandas as pd
from sklearn.neighbors import KernelDensity
import numpy as np

class TADE:
    def __init__(self, full_cartesian=False, **kwargs):
        self.kernel = 'gaussian'
        self.model = None
        self.max_t = 0.0
        self.full_cartesian = full_cartesian

    def train(self, log_df):
        d = {}
        log_df['time:timestamp'] = pd.to_datetime(log_df['time:timestamp'])
        grouped = log_df.groupby('case:concept:name')

        for case_id, group in grouped:
            group = group.sort_values('time:timestamp')
            times = group['time:timestamp']
            tmin = times.min()

            if self.full_cartesian:
                for i, e1 in group.iterrows():
                    a1 = e1['concept:name']
                    t1 = e1['time:timestamp']
                    for j, e2 in group.iterrows():
                        a2 = e2['concept:name']
                        t2 = e2['time:timestamp']
                        std_t = (t2 - t1).total_seconds()
                        tup = f"{a1}->{a2}"
                        if tup in d:
                            d[tup].append(std_t)
                        else:
                            d[tup] = [std_t]

                        if std_t > self.max_t:
                            self.max_t = std_t
            else:
                for i, e in group.iterrows():
                    act = e['concept:name']
                    t = e['time:timestamp']
                    std_t = (t - tmin).total_seconds()

                    if act in d:
                        d[act].append(std_t)
                    else:
                        d[act] = [std_t]

                    if std_t > self.max_t:
                        self.max_t = std_t

        self.model = {}
        for act, value in d.items():
            h = np.std(value) * (4 / 3 / len(value)) ** (1 / 5)
            if h == 0.0:
                h = 1.0
            self.model[act] = KernelDensity(kernel=self.kernel, bandwidth=h).fit(np.array(value).reshape(-1, 1))
    def fitness(self, log_df):
        temp_values = {}
        log_df['time:timestamp'] = pd.to_datetime(log_df['time:timestamp'])
        
        grouped = log_df.groupby('case:concept:name')  # Group by traces
        
        if self.full_cartesian:
            d = {}
            for case_id, trace_df in grouped:
                trace_df = trace_df.sort_values('time:timestamp')
                for i, e1 in trace_df.iterrows():
                    a1 = e1['concept:name']
                    t1 = e1['time:timestamp']
                    for j, e2 in trace_df.iterrows():
                        a2 = e2['concept:name']
                        t2 = e2['time:timestamp']
                        std_t = (t2 - t1).total_seconds()
                        tup = f"{a1}->{a2}"
                        if tup in d:
                            d[tup].append(std_t)
                        else:
                            d[tup] = [std_t]
            for act, value in d.items():
                temp_values[act] = np.mean(d[act])
        else:
            d = {}
            for case_id, trace_df in grouped:
                tmin = trace_df['time:timestamp'].min()
                for i, e in trace_df.iterrows():
                    act = e['concept:name']
                    t = e['time:timestamp']
                    std_t = (t - tmin).total_seconds()
                    if act in d:
                        d[act].append(std_t)
                    else:
                        d[act] = [std_t]
            for act, value in d.items():
                temp_values[act] = np.mean(d[act])

        conf_scores = []
        for act, t in temp_values.items():
            if act in self.model:
                y = self.model[act].score_samples([[t]])
                conf_scores.append(y[0])
            else:
                conf_scores.append(0)
        
        result = np.mean(conf_scores)
        return np.exp(result)