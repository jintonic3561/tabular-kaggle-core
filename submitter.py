# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 08:05:09 2023

@author: jintonic
"""

import pandas as pd
import numpy as np
import os
from collections import namedtuple
from abstract import CodeSubmitter


class AveragingSubmitter(CodeSubmitter):
    pred_col = 'pred'
    target_col = 'y'
    
    def __init__(self, submitters, csv_dir, thr_dir, submission_comment):
        self.submitters = submitters
        self.submission_comment = submission_comment
        
    def experiment(self,
                   params={},
                   retrain_all_data: bool=False,
                   dry_run: bool=False,
                   return_only: bool=False):
        # Note: 互換性のための引数であり無意味
        res = self._train_and_evaluate(retrain_all_data=retrain_all_data,
                                       dry_run=dry_run)
        if not dry_run:
            if return_only:
                return res
            else:
                experiment_params = {**self.get_experiment_params(), **params}
                self._save_experiment(self.get_metrics(res), params=experiment_params)
        else:
            breakpoint()
            
    def load_model(self):
        for i in self.submitters:
            i.model.load_model()
        
    def estimate(self, test, sub):
        preds = []
        for i in self.submitters:
            sub = i.estimate(test=test.copy(), sub=sub.copy(), proba=True)
            preds.append(sub[self.target_col].values)
            
        preds = np.stack(preds)
        average_pred = preds.mean(axis=0)
        sub[self.target_col] = average_pred
        return sub
    
    def get_experiment_params(self):
        params = {'model_name': 'ensemble'}
        for index, submitter in enumerate(self.submitters):
            params[f'element_{index}'] = submitter.model.__class__.__name__
        return params
    
    def _train_and_evaluate(self, retrain_all_data=None, dry_run=None):
        oof = None
        pred_cols = []
        for i, submitter in enumerate(self.submitters):
            temp = self._load_csv(submitter.model.csv_dir)
            pred_col = f'pred_{i}'
            pred_cols.append(pred_col)
            if oof is None:
                oof = temp.rename(columns={'pred': pred_col})
            else:
                temp = temp[['session', 'question', 'pred']].rename(columns={'pred': pred_col})
                oof = pd.merge(oof, temp, how='inner', on=['session', 'question'])
        
        oof['pred'] = oof[pred_cols].mean(axis=1)
        oof = oof.drop(columns=pred_cols)
        submitter.model._save_oof_pred(oof)
        metrics = submitter.model._calc_metric(oof)
        
        if submitter.model.regression:
            Result = namedtuple('Result', ['metrics', 'cv_preds', 'permutaion_importance'])
            return Result(metrics, oof, None)
        else:
            clf_metrics = self.calc_classification_metrics(oof)
            Result = namedtuple('Result', ['learn_metrics', 'clf_metrics', 'cv_preds', 'permutaion_importance'])
            return Result(metrics, clf_metrics, oof, None) 
                
    def _load_csv(self, oof_dir):
        path = os.path.join(oof_dir, 'cv_preds.csv')
        return pd.read_csv(path)



class StackingSubmitter(AveragingSubmitter):
    id_col = ''
    
    def __init__(self, stack_submitter, submitters, submission_comment):
        self.stack_submitter = stack_submitter
        self.submitters = submitters
        self.submission_comment = submission_comment
        assert self.id_col
        
    def load_model(self):
        super().load_model()
        self.stack_submitter.load_model()
    
    def estimate(self, test, sub):
        features = self._estimate_layer_0(test=test, sub=sub)
        sub = self._estimate_layer_1(features=features, sub=sub)
        return sub
    
    def _estimate_layer_0(self, test, sub):
        features = sub[[self.id_col]].copy()
        for index, submitter in enumerate(self.submitters):
            temp = submitter.estimate(test=test.copy(), sub=sub.copy(), proba=True)
            temp = temp.rename(columns={self.target_col: f'pred_{index}'})
            features = pd.merge(features, temp, how='left', on=self.id_col)
        return features
    
    def _estimate_layer_1(self, features, sub):
        pred = self.stack_submitter.model.estimate(features, sub.copy(), proba=False)
        sub = pd.merge(sub, pred, how='left', on=self.id_col)
        return sub
    
    def get_experiment_params(self):
        params = {'model_name': self.stack_submitter.model.__class__.__name__}
        for index, submitter in enumerate(self.submitters):
            params[f'element_{index}'] = submitter.model.__class__.__name__
        return params
    
    def _train_and_evaluate(self, retrain_all_data=False, dry_run=None):
        features = self._generate_features()
        fold_generate_func = self.stack_submitter.data_splitter.cv_split
        res = self.stack_submitter.model.cv(features=features,
                                            fold_generate_func=fold_generate_func,
                                            save_model=not retrain_all_data)
        if retrain_all_data:
            self.stack_submitter.model.fit(features=features, save_model=True)
        return res
    
    def _generate_features(self):
        features = pd.DataFrame()
        for index, submitter in enumerate(self.submitters):
            pred_col = f'pred_{index}'
            df = self._load_csv(submitter.model.csv_dir)
            df = df.rename(columns={self.pred_col: pred_col})
            if index == 0:
                features = df.copy()
            else:
                df = df[[self.id_col, pred_col]]
                features = pd.merge(features, df, how='inner', on=self.id_col)
        
        # features = self._calc_agg_features(features)
        return features
    
    # def _calc_agg_features(self, df):
    #     pred_cols = [i for i in df.columns if 'pred' in i]
    #     # df['pred_mean'] = df[pred_cols].mean(axis=1)
    #     df['pred_std'] = df[pred_cols].std(axis=1)
    #     df['pred_min'] = df[pred_cols].min(axis=1)
    #     df['pred_max'] = df[pred_cols].max(axis=1)
    #     # df['pred_range'] = df['pred_max'] - df['pred_min']
    #     return df






