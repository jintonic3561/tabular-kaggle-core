import os

def set_config(local=False, experiment=False, cloud=False):
    if experiment and cloud:
        # Note: for Vertex AI Workbench
        dataset_root_dir = os.path.join("/home/jupyter/imported/input/", os.environ["KAGGLE_DATASET_NAME"])
    else:
        dataset_root_dir = os.path.join("/kaggle/input/", os.environ["KAGGLE_DATASET_NAME"])
    
    if not local and experiment and not cloud:
        os.environ["DATASET_ROOT_DIR"] = "/kaggle/working/"
    else:
        os.environ["DATASET_ROOT_DIR"] = dataset_root_dir

    if experiment or local:
        import pretty_errors

        pretty_errors.configure(
            display_locals=True,
            filename_display=pretty_errors.FILENAME_EXTENDED,
        )

    if not experiment:
        import numpy as np

        np.seterr(divide="ignore", invalid="ignore")
