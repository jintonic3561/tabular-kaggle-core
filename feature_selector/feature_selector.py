# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 15:04:22 2023

@author: jintonic
"""


import pandas as pd
import time
from abstract import ABSSubmitter


class BottomUpFeatureSelector:
    def __init__(self, 
                 submitter: ABSSubmitter,
                 base_features,
                 header_columns,
                 checkpoint_path=None,
                 output_path='./feature_selection_result.csv',
                 time_limit=60*60*10):
        self.submitter = submitter
        self.base_features = base_features
        self.header_columns = header_columns
        self.checkpoint_path = checkpoint_path
        self.output_path = output_path
        self.time_limit = time_limit
        
    def run(self, dry_run=False, data_process_id=None):
        self.preparate_experiment(dry_run=dry_run, data_process_id=data_process_id)
        for candidate in self.candidates:
            metric = self.experiment(candidate)
            adopt = self.logger(candidate, metric)
            if adopt:
                self.selected.append(candidate)
            if self.is_time_limit():
                break
        result = self.logger.result
        self.save_experiment(result)
        return result
        
    def experiment(self, candidate):
        if candidate:
            features = self.features[self.header_columns + self.selected + [candidate]]
        else:
            features = self.features[self.header_columns + self.selected]
            
        res = self.submitter._train_and_evaluate(features, 
                                                 retrain_all_data=False,
                                                 save_model=False)
        self.elapsed_time.append(time.time())
        return self.get_metric(res)
    
    def preparate_experiment(self, dry_run, data_process_id):
        self.elapsed_time = [time.time()]
        self.features = self.data_fetcher(dry_run=dry_run)
        self.features = self._process_data(memory_id=data_process_id)
        self.logger = BottomUpLogger()
        if self.checkpoint_path:
            adopted, rejected = self.logger.load_checkopoint(self.checkpoint_path)
            self.base_features += adopted
            self.selected = self.base_features
            ignore_cols = rejected
        else:
            self.selected = self.base_features
            baseline_score = self.experiment(None)
            self.logger.set_baseline(baseline_score)
            ignore_cols = []
        
        ignore_cols += self.submitter.model.base_model.ignore_columns + self.base_features
        self.candidates = [i for i in self.features.columns if i not in ignore_cols]
        
    def get_metric(self, res):
        return res.f1
    
    def save_experiment(self, result):
        result.to_csv(self.output_path, index=False)
        
    def is_time_limit(self):
        if len(self.elapsed_time) < 2:
            return False
        next_time = time.time() - self.elapsed_time[0] + self.elapsed_time[-1] - self.elapsed_time[-2]
        return next_time > self.time_limit



class BottomUpLogger:
    def __init__(self, adopt_thr=0.00005):
        self.thr = adopt_thr
        self.sota = 0.0
        self.result = pd.DataFrame([], columns=['feature', 'metric', 'gain', 'adopt'])
        
    def __call__(self, candidate, metric):
        metric = round(metric, 5)
        gain = round(metric - self.sota, 5)
        adopt = False
        if gain > self.thr:
            self.sota = metric
            adopt = True
        
        row = {'feature': candidate,
               'metric': metric,
               'gain': gain,
               'adopt': adopt}
        self.result = self.result.append(row, ignore_index=True)
        print(row)
        print('----------')
        return adopt
    
    def set_baseline(self, score):
        self.sota = round(score, 5)
        
    def load_checkopoint(self, path):
        self.result = pd.read_csv(path)
        self.set_baseline(self.result['metric'].max())
        adopted = self.result[self.result['adopt']]['feature'].to_list()
        rejected = self.result[~self.result['adopt']]['feature'].to_list()
        return adopted, rejected









