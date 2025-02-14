# Description: This file contains the implementation of the TADE conformance checking algorithm.
# Code obtained from: https://github.com/Skarvir/TADE


from sklearn.neighbors import KernelDensity
import numpy as np
from pm4py.objects.log.importer.xes import factory as xes_importer

class TADE:
    def __init__(self, full_cartesian = False, **kwargs):
        self.kernel = 'gaussian'
        self.model = None
        self.max_t = 0.0
        self.full_cartesian = False
        
    
    def train(self, log):
        d = {}
        for trace in log:
            times = [e['time:timestamp'] for e in trace]
            tmin = min(times)
            
            if self.full_cartesian:
                for e1 in trace:
                    a1 = e1['concept:name']
                    t1 = e1['time:timestamp']
                    for e2 in trace:
                        a2 = e2['concept:name']
                        t2 = e2['time:timestamp']
                        std_t = (t2 - t1).total_seconds()
                        tup = a1+"->"+a2
                        if tup in d:
                            d[tup].append(std_t)
                        else:
                            d[tup] = [std_t]                
                          
                        if std_t > self.max_t:
                            self.max_t = std_t
                
            else:
                for e in trace:
                    act = e['concept:name']
                    t = e['time:timestamp']
        
                    std_t =  (t - tmin).total_seconds()
                    
                    if act in d:
                        d[act].append(std_t)
                    else:
                        d[act] = [std_t]
                       
                        
                    if std_t > self.max_t:
                        self.max_t = std_t
        self.model = {}    
        for act, value in d.items():
            
            # Silvermans rule of thumb
            h = np.std(value)*(4/3/len(value))**(1/5)
            if h == 0.0:
                h = 1.0
            self.model[act] = KernelDensity(kernel=self.kernel, bandwidth=h).fit(np.array(value).reshape(-1,1))
        
        
    def fitness(self, trace):
        temp_values = {}
        if self.full_cartesian:
            d = {}
            for e1 in trace:
                a1 = e1['concept:name']
                t1 = e1['time:timestamp']
                for e2 in trace:
                    a2 = e2['concept:name']
                    t2 = e2['time:timestamp']
                    std_t = (t2 - t1).total_seconds()
                    tup = a1+"->"+a2
                    if tup in d:
                        d[tup].append(std_t)
                    else:
                        d[tup] = [std_t]                
            for act, value in d.items():
                temp_values[act] = np.mean(d[act])
        
        else:
            tmin = min([e['time:timestamp'] for e in trace])
            d = {}
            for e in trace:
                act = e['concept:name']
                t = e['time:timestamp']
                std_t =  (t - tmin).total_seconds()
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
        result = (np.mean(conf_scores))
        return np.exp(result)

