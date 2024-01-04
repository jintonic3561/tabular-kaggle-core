from source.worker import *

config = {
    "experiment_name": "baseline",
    "feature_experiment_name": "baseline",
    "model_experiment_name": "baseline",
    "experiment_params": {},
}


def get_submitter():
    raise NotImplementedError("Implement a function that return a submitter")
