import importlib
import os
import time

import pandas as pd
from source.abs.config import set_config


def run(
    ex_num, local=False, dry_run=False, ignore_exception=False
):
    start_time = time.time()
    set_config(local=local)
    submitter, _ = _get_submitter(ex_num)

    if local:
        env = submitter.get_mock_api(dry_run=dry_run)
        iter_test = env.iter_test()
        # Note: Validate inferences using only the last fold's model
        submitter.set_last_fold_model()
    else:
        iter_test = _get_iter_test_object()

    iter_times = []
    for index, *args in enumerate(iter_test):
        iter_start = time.time()
        sub = submitter.estimate(*args, ignore_exception=ignore_exception)
        env.predict(sub)

        iter_time = time.time() - iter_start
        iter_times.append(iter_time)
        print(f"\riter: {index}  iter_time: {iter_time:.3f}[s], ", end="")

    print(f"total_time: {time.time() - start_time:.3f}[s]")
    print(f"average_time: {sum(iter_times) / len(iter_times):.3f}[s]")
    print(f"iteration count: {len(iter_times)}")


def _get_submitter(ex_num):
    ex = importlib.import_module(
        f"experiment_setting.ex_{ex_num}", package="experiment_setting"
    )
    submitter = ex.get_submitter()
    submitter.seed_everything()
    submitter.load_model()
    return submitter, ex.config


def _get_iter_test_object():
    raise NotImplementedError("Implement a function that returns an iter_test object in the submit environment.")


def _validate_prediction(ex_num, id_columns, error_threshold=1e-6, oof_dir=None, fold_id=-1):
    submitter, config = _get_submitter(ex_num)
    sub = pd.read_csv(
        os.path.join(os.environ["DATASET_ROOT_DIR"], "artifact/temp/submission.csv")
    )
    oof = pd.read_csv(
        os.path.join(
            os.environ["DATASET_ROOT_DIR"],
            "artifact/oof_pred",
            oof_dir if oof_dir else config["model_experiment_name"],
            "oof_pred.csv",
        )
    )
    fold_id = oof["cv_id"].max() if fold_id == -1 else fold_id
    target_col = submitter.model.target_col
    oof[oof["cv_id"] == fold_id][id_columns + ["pred", target_col]]
    oof = oof.rename(columns={"pred": "oof_pred"})
    sub = sub.rename(columns={target_col: "sub_pred"})
    df = pd.merge(oof, sub, on=id_columns, how="left")

    assert df.isnull().sum().sum() == 0
    oof_score = submitter.model._calc_metric(
        df[[target_col, "oof_pred"]].rename(columns={"oof_pred": "pred"})
    )
    sub_score = submitter.model._calc_metric(
        df[[target_col, "sub_pred"]].rename(columns={"sub_pred": "pred"})
    )
    print(f"oof_score: {oof_score}, sub_score: {sub_score}")
    assert abs(oof_score - sub_score) < error_threshold
    print("Validation passed!")
    

def _upload_dataset(ex_num):
    dataset_title = os.environ["KAGGLE_DATASET_NAME"]
    dataset_directory = os.path.join(
        os.path.join(os.environ["DATASET_ROOT_DIR"], "artifact/dataset/")
    )
    submitter, _ = _get_submitter(ex_num)
    res = submitter.upload_dataset(dataset_title=dataset_title, dataset_directory=dataset_directory)
    print(res)


if __name__ == "__main__":
    ex_num = 0
    dry_run = True
    ignore_exception = False
    id_columns = ["not_implement_error"]
    set_config(local=True)
    
    run(
        ex_num,
        local=True,
        dry_run=dry_run,
        ignore_exception=ignore_exception,
    )
    _validate_prediction(ex_num, id_columns=id_columns)

    # _upload_dataset(ex_num)
