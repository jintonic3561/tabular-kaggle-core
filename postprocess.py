# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:18:50 2023

@author: jintonic
"""

from sklearn.preprocessing import QuantileTransformer
from abstract import ABSDataPostprocessor
SEED = 42


class RankGaussPostProcessor(ABSDataPostprocessor):
    def __init__(self, 
                 ignore_columns=['session', 'level_group'],
                 save_dir='./data/rankgauss_processor/default/',
                 infer=False,
                 processor_path=None):
        super().__init__(save_dir=save_dir)
        self.ignore_columns = ignore_columns
        self.infer = infer
        self.processor_path = processor_path
        if infer:
            assert processor_path
        
    def main(self, df):
        if self.infer:
            processor = self.load(self.processor_path)
        else:
            processor = QuantileTransformer(random_state=SEED, output_distribution='normal')
            processor.fit(df.drop(columns=self.ignore_columns).values)
            self.save(processor)
        
        transform_cols = [i for i in df.columns if i not in self.ignore_columns]
        df[transform_cols] = processor.transform(df[transform_cols].values)
        return df