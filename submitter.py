# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 08:05:09 2023

@author: jintonic
"""

import itertools
import os
import time
from collections import namedtuple

import numpy as np
import optuna
import pandas as pd
from tqdm import tqdm

try:
    from abstract import ABSSubmitter, CodeSubmitter
except ModuleNotFoundError:
    from .abstract import ABSSubmitter, CodeSubmitter


class AveragingSubmitter(ABSSubmitter):
    cv_id_col = "cv_id"
    pred_col = "pred"
    target_col = "y"

    def __init__(
        self,
        cv_paths,
        sub_paths,
        model,
        submission_comment,
        submission_csv_dir="./submission/default/",
    ):
        self.cv_paths = cv_paths
        self.sub_paths = sub_paths
        super().__init__(
            data_fetcher=None,
            data_preprocessor=None,
            feature_generator=None,
            data_splitter=None,
            data_postprocessor=None,
            model=model,
            submission_comment=submission_comment,
            submission_csv_dir=submission_csv_dir,
        )

    def make_submission(self, experiment_params=None):
        cv_metrics = self._evalute()
        sub = self._get_submit_data()
        self._submit(sub)
        time.sleep(15)
        self._save_experiment(cv_metrics, params=experiment_params)

    def _evalute(self):
        temp = self._load_csv(self.cv_paths[0])
        cv_ids = temp[self.cv_id_col].unique()
        cv = []
        metrics = []
        for id in cv_ids:
            cv.append(temp[temp[self.cv_id_col] == id])
        for path in self.cv_paths[1:]:
            temp = self._load_csv(path)
            for index, id in enumerate(cv_ids):
                cv[index][self.pred_col] += temp[temp[self.cv_id_col] == id][
                    self.pred_col
                ]
        for index in range(len(cv)):
            cv[index][self.pred_col] /= len(self.cv_paths)
            metrics.append(self.model._calc_metric(cv[index]))
        return metrics

    def _get_submit_data(self):
        sub = self._load_csv(self.sub_paths[0])
        for path in self.sub_paths[1:]:
            temp = self._load_csv(path)
            sub[self.target_col] += temp[self.target_col]
        sub[self.target_col] /= len(self.sub_paths)
        return sub

    def _load_csv(self, path):
        return pd.read_csv(path)


class CodeBlendingSubmitter(CodeSubmitter):
    pred_col = "pred"
    target_col = "y"

    def __init__(
        self,
        submitters,
        merge_keys,
        submission_comment,
        experiment_params={},
        weights=None,
        optimize=False,
        optimize_method="grid",
        optimize_direction="minimize",
        n_trials=100,
        search_grid=None,
    ):
        """
        defaultはAveraging
            - weights=None and optimize=False ならAveraging
            - weights=None and optimize=True ならBlending探索
            - weights!=NoneならBlending固定

        探索方法はgrid or optuna
        gridの場合は、カスタムのsearch_gridを指定可能
        """

        self.submitters = submitters
        self.merge_keys = merge_keys
        self.submission_comment = submission_comment
        self.experiment_params = experiment_params

        if weights is None and not optimize:
            self.weights = [1 / len(submitters) for _ in range(len(submitters))]
        else:
            self.weights = weights
        self.optimize = optimize
        self.optimize_method = optimize_method
        self.optimize_direction = optimize_direction
        self.n_trials = n_trials
        self.search_grid = search_grid

        # dummy
        self.model = self.submitters[0].model

        if not self.competition_name:
            raise ValueError("competition_name must be specified.")
        if not self.experiment_name:
            raise ValueError("experiment_name must be specified.")

        self.api = self._init_kaggle_api()

    def experiment(
        self,
        params={},
        # Note: 互換性のための引数であり無意味
        retrain_all_data: bool = False,
        dry_run: bool = False,
        return_only: bool = False,
    ):
        res = self._train_and_evaluate()
        if not dry_run:
            if return_only:
                return res
            else:
                self._save_experiment(res, params=self.get_experiment_params())
        else:
            breakpoint()

    def load_model(self):
        for i in self.submitters:
            i.model.load_model()

    def estimate(self, test, sub):
        """
        Note: 特徴量作成が重複するため、コンペ設計に合わせて実装する
        """
        raise NotImplementedError

    def get_experiment_params(self):
        params = {"model": "ensemble"}
        for index, submitter in enumerate(self.submitters):
            params[f"element_{index}"] = submitter.model.__class__.__name__
        for index, weights in enumerate(self.weights):
            params[f"weight_{index}"] = weights
        return params

    def _train_and_evaluate(self):
        # Note: optunaの場合のobjectiveに対応するためAttributeにする
        self.oof = self._merge_oof_predition()
        if self.weights is None:
            self._optimize_weights()

        oof = self._blending(df=self.oof, weights=self.weights)
        self._save_oof(oof)
        res = self._calc_metric(oof)
        return res

    def _blending(self, df, weights):
        temp = df.copy()
        temp["pred"] = 0.0
        pred_cols = self._get_pred_cols()
        for w, c in zip(weights, pred_cols):
            temp["pred"] += w * temp[c]
        temp = temp.drop(columns=pred_cols)
        return temp

    def _merge_oof_predition(self):
        oof = None
        for i, submitter in enumerate(self.submitters):
            temp = self._load_oof(submitter.model.oof_dir)
            pred_col = f"{self.pred_col}_{i}"
            if oof is None:
                oof = temp.rename(columns={self.pred_col: pred_col})
            else:
                temp = temp[self.merge_keys + [self.pred_col]].rename(
                    columns={self.pred_col: pred_col}
                )
                oof = pd.merge(oof, temp, how="inner", on=self.merge_keys)
                assert len(oof) == len(temp)
        return oof

    def _get_pred_cols(self):
        return [f"{self.pred_col}_{i}" for i in range(len(self.submitters))]

    def _calc_metric(self, oof):
        metrics = []
        for i in oof["cv_id"].unique():
            temp = oof[oof["cv_id"] == i]
            metrics.append(self.submitters[0].model._calc_metric(temp))

        if self.submitters[0].model.regression:
            oof_metrics = self.submitters[0].model._calc_metric(oof)
        else:
            oof_metrics = self.submitters[0].model.calc_classification_metrics(oof)

        Result = namedtuple(
            "Result",
            ["cv_metrics", "oof_metrics", "cv_preds", "permutaion_importance"],
        )
        return Result(metrics, oof_metrics, oof, None)

    def _load_oof(self, oof_dir):
        path = os.path.join(oof_dir, "oof_pred.csv")
        return pd.read_csv(path)

    def _save_oof(self, oof):
        path = os.path.join(
            os.environ["DATASET_ROOT_DIR"],
            "artifact/oof_pred/",
            self.submission_comment,
            "oof_pred.csv",
        )
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        oof.to_csv(path, index=False)

    def _optimize_weights(self):
        if self.optimize_method == "grid":
            self.weights = self._optimize_with_grid()
        elif self.optimize_method == "optuna":
            self.weights = self._optimize_with_optuna()
        else:
            raise ValueError("optimize_method must be grid or optuna.")

    def _get_objective(self, cv_result) -> float:
        return round(np.mean(cv_result.cv_metrics), 4)

    def _optimize_with_grid(self):
        weights_list = []
        scores = []
        if self.search_grid is None:
            self.search_grid = self._get_params_grid()
        for weights in tqdm(self.search_grid):
            temp = self._blending(df=self.oof, weights=weights)
            res = self._calc_metric(temp)
            score = self._get_objective(res)
            weights_list.append(weights)
            scores.append(score)

        best_weights = weights_list[np.argmin(scores)]
        print("Best weights:", best_weights, "Best score:", np.min(scores))
        return best_weights

    def _get_params_grid(self, unit=0.05):
        # 3つのパラメータのすべての組み合わせを作成
        grid = list(
            itertools.product(np.arange(0, 1.1, unit), repeat=len(self.submitters))
        )
        # 各組み合わせの合計が1になる組み合わせだけをフィルタリング
        grid = [i for i in grid if np.isclose(sum(i), 1, atol=unit / 10)]
        return grid

    def _optuna_objective(self, trial):
        model_num = len(self.submitters)
        weights = [trial.suggest_float(f"w_{i}", 0, 1) for i in range(model_num)]
        weights = np.array(weights) / np.sum(weights)
        temp = self._blending(df=self.oof, weights=weights)
        res = self._calc_metric(temp)
        score = self._get_objective(res)
        return score

    def _optimize_with_optuna(self):
        study = optuna.create_study(direction=self.optimize_direction)
        study.optimize(self._optuna_objective, n_trials=self.n_trials)
        best_weights = study.best_trial.params
        print("Best weights:", best_weights)
        return list(best_weights.values())

    def set_last_fold_model(self):
        for i in self.submitters:
            i.set_last_fold_model()


class CodeStackingSubmitter(CodeBlendingSubmitter):
    def __init__(
        self,
        layer_1_submitter,
        layer_0_submitters,
        merge_keys,
        submission_comment,
        experiment_params={},
        add_agg_features=False,
    ):
        self.layer_1_submitter = layer_1_submitter
        self.add_agg_features = add_agg_features
        super().__init__(
            submitters=layer_0_submitters,
            merge_keys=merge_keys,
            submission_comment=submission_comment,
            experiment_params=experiment_params,
        )

    def load_model(self):
        super().load_model()
        self.layer_1_submitter.load_model()

    def estimate(self, test, sub):
        features = self._estimate_layer_0(test=test, sub=sub)
        sub = self._estimate_layer_1(features=features, sub=sub)
        return sub

    def _estimate_layer_0(self, test, sub):
        features = sub[self.merge_keys].copy()
        for index, submitter in enumerate(self.layer_0_submitters):
            temp = submitter.estimate(test=test.copy(), sub=sub.copy())
            temp = temp.rename(columns={self.target_col: f"pred_{index}"})
            features = pd.merge(features, temp, how="left", on=self.id_col)
        return features

    def _estimate_layer_1(self, features, sub):
        pred = self.layer_1_submitter.model.estimate(features, sub.copy())
        sub = pd.merge(sub, pred, how="left", on=self.merge_keys)
        return sub

    def _train_and_evaluate(self):
        features = self._get_oof_features()
        fold_generator = self._cv_split(features)
        res = self.layer_1_submitter.model.cv(fold_generator, save_model=True)
        return res

    def _get_oof_features(self):
        df = self._merge_oof_predition()
        if self.add_agg_features:
            df = self._calc_agg_features(df)
        return df

    def _calc_agg_features(self, df):
        pred_cols = [i for i in df.columns if self.pred_col in i]
        df[f"{self.pred_col}_mean"] = df[pred_cols].mean(axis=1)
        df[f"{self.pred_col}_std"] = df[pred_cols].std(axis=1)
        df[f"{self.pred_col}_min"] = df[pred_cols].min(axis=1)
        df[f"{self.pred_col}_max"] = df[pred_cols].max(axis=1)
        df[f"{self.pred_col}_range"] = (
            df[f"{self.pred_col}_max"] - df[f"{self.pred_col}_min"]
        )
        return df

    def get_experiment_params(self):
        params = {"model": self.layer_1_submitter.model.__class__.__name__}
        for index, submitter in enumerate(self.layer_0_submitters):
            params[f"element_{index}"] = submitter.model.__class__.__name__
        return params

    def _cv_split(self, features):
        for i in features["cv_id"].unique():
            train = features[features["cv_id"] != i]
            valid = features[features["cv_id"] == i]
            yield train, valid
