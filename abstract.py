# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 07:10:20 2023

@author: jintonic
"""


import pandas as pd
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
from mlutil.util import mlflow
from mlutil.features import ABSFeatureGenerator
from mlutil.mlbase import MLBase



class ABSDataFetcher:
    data_dir = './data/'
    
    def __init__(self):
        pass
    
    def __call__(self, dry_run: bool=False) -> pd.DataFrame:
        return self._load_data(dry_run=dry_run)
    
    def _load_data(self, dry_run: bool):
        raise NotImplementedError()



class ABSDataPreprocessor:
    def __init__(self):
        pass
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()



def init_preprocessor(*args):
    def _apply(df):
        for processor in args:
            df = processor(df)
        return df
    return _apply
    


class ABSDataPostprocessor:
    def __init__(self):
        pass
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()



class ABSDataSplitter:
    def __init__(self):
        pass
    
    def train_test_split(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
        
    def cv_split(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Parameters
        ----------
        df : pd.DataFrame
        
        Yields
        -------
        pd.DataFrame
        '''
        raise NotImplementedError()
        


class ABSModel:
    def __init__(self, cv_method: str, cv_folds: int, **kwargs):
        # TODO: mlbaseのリファクタ
        '''
        1. kwargsをなんとかする
        2. cvを何とかする
            ・cvの引数にgenerator(yield train, valid)を持てるようにし、model内にCV分割関数を持たないようにする。
            ・cv内でモデルの保存を行い、各モデルのvalid予測結果をreturnできるようにする。
        '''        
        self.cv_method = cv_method
        self.cv_folds = cv_folds
        self.cv_predictions = []



class ABSPredPostProcessor:
    def __init__(self):
        pass
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()



class ABSSubmitter:
    data_dir = './data/'
    competition_name = ''
    
    def __init__(self, 
                 data_fetcher: ABSDataFetcher,
                 data_preprocessor: ABSDataPreprocessor,
                 feature_generator: ABSFeatureGenerator,
                 data_splitter: ABSDataSplitter,
                 model: MLBase,
                 submission_comment: str,
                 data_postprocessor: ABSDataPostprocessor=None,
                 pred_postprocessor: ABSPredPostProcessor=None):
        '''
        Parameters
        ----------
        data_fetcher: ABSDataFetcher
        data_preprocessor: ABSDataPreprocessor
        feature_generator: ABSFeatureGenerator
        data_splitter: ABSDataSplitter
        model: ABSModel, MLBase
        submission_comment: str
            The Message for submission.
        pred_postprocessor: ABSPredPostProcessor, optional
        data_postprocessor: ABSDataPostprocessor, optional
        '''
        
        if not self.competition_name :
            raise ValueError('competition_name must be specified.')
        
        self.data_fetcher = data_fetcher
        self.data_preprocessor = data_preprocessor
        self.feature_generator = feature_generator
        self.data_splitter = data_splitter
        self.model = model
        self.submission_comment = submission_comment
        self.data_postprocessor = data_postprocessor
        self.pred_postprocessor = pred_postprocessor
        self.api = self._init_kaggle_api()
    
    def _get_submit_data(self, test: pd.DataFrame) -> pd.DataFrame:
        # TODO: mlbaseをリファクタしcv averagingを実装
        raise NotImplementedError()
    
    def make_submission(self, dry_run: bool=False):
        data = self._process_data(dry_run=dry_run)
        train, test = self.data_splitter.train_test_split(data)
        metrics = self._train_and_evaluate(train)
        sub = self._get_submit_data(test)
        if self.pred_postprocessor:
            sub = self.pred_postprocessor(sub)
        if not dry_run:
            self._submit(sub)
            self._save_experiment(metrics)
        else:
            breakpoint()
            
    def _process_data(self, dry_run: bool):
        data = self.data_fetcher(dry_run=dry_run)
        data = self.data_preprocessor(data)
        data = self.feature_generator(data)
        if self.data_postprocessor:
            data = self.data_postprocessor(data)
        return data
    
    def _train_and_evaluate(self, 
                            train: pd.DataFrame, 
                            retrain_all_data: bool=False,
                            save_model: bool=True) -> list:
        cv_generator = self.data_splitter.cv_split(train)
        # TODO: cv_predictionを返り値として受け取る
        res = self.model.cv(train, cv_generator=cv_generator)
        if retrain_all_data:
            self.model.fit(train, save_model=save_model)
        return res.metrics
    
    def _submit(self, test: pd.DataFrame):
        file_name = f'{self.data_dir}submission.csv'
        test.to_csv(file_name, index=False)
        self.api.competition_submit(file_name=file_name,
                                    message=self.submission_comment,
                                    competition=self.competition_name)
    
    def _init_kaggle_api(self) -> KaggleApi:
        api = KaggleApi()
        api.authenticate()
        return api
    
    def _get_public_score(self) -> float:
        sub = self.api.competitions_submissions_list(self.competition_name)
        sub = pd.DataFrame(sub)
        sub['date'] = pd.to_datetime(sub['date'])
        score = sub.sort_values('date', ascending=False)['publicScoreNullable'].iloc[0]
        score = float(score) if score else np.nan
        return score
    
    def _save_experiment(self, cv_metrics: list, params: dict):
        mean = np.array(cv_metrics).mean()
        std = np.array(cv_metrics).std()
        sharpe = self._calc_sharpe(mean, std)
        public_score = self._get_public_score()
        experiment_name = self.competition_name.split('-')[0]
        mlflow.run(experiment_name=experiment_name,
                   run_name=self.submission_comment,
                   params=params,
                   metrics={'cv_mean': mean,
                            'cv_std': std,
                            'cv_sharpe': sharpe,
                            'public_score': public_score},
                   artifact_paths=[self.model.model_path])
        print(f'CV metrics: {[round(i, 4) for i in cv_metrics]}')
        print(f'mean: {round(mean, 4)}, std: {round(std, 4)}, sharpe: {round(sharpe, 4)}')
        
    def _calc_sharpe(self, mean, std):
        return mean / (std + 1)
    
    



