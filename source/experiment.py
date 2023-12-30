import importlib
import os

from source.abs.config import set_config
from source.mlutil import tuner as Tuner


def run(ex_num, dry_run=False, local=True, cloud=False):
    submitter = _get_submitter(ex_num, dry_run=dry_run, local=local, cloud=cloud)
    submitter.experiment(dry_run=dry_run)


def tuning(ex_num, model_class, n_trials=100, dry_run=False, local=True, cloud=False):
    submitter = _get_submitter(ex_num, dry_run=dry_run, local=local, cloud=cloud)
    tuner = Tuner.LGBTuner(
        submitter=submitter,
        model_class=model_class,
        n_trials=n_trials,
        dry_run=dry_run,
        initial_trial_params=Tuner.initial_trial_params,
    )
    return tuner.run()


def _get_submitter(ex_num, dry_run=False, local=True, cloud=False):
    if dry_run:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    set_config(local=local, experiment=True, cloud=cloud)
    ex = importlib.import_module(
        f"experiment_setting.ex_{ex_num}", package="experiment_setting"
    )
    submitter = ex.get_submitter()
    submitter.seed_everything()
    return submitter



if __name__ == "__main__":
    ex_num = 0
    dry_run = True
    local = True

    run(ex_num, dry_run=dry_run, local=local)