import json
import argparse


def main():
    args = _load_args()
    _update_json(args)

def _load_args():
    parser = argparse.ArgumentParser(description="Confiure your container settings")
    parser.add_argument("--competition_id", default="kaggle", help="your project name")
    parser.add_argument("--ds_name", default="dataset", help="Title of dataset to be uploaded to kaggle")
    parser.add_argument("--device", default="gpu", help="gpu or cpu")
    parser.add_argument("--debug", default=False, help="true or false")
    args = parser.parse_args()
    return args

def _update_json(args):
    with open('.devcontainer/devcontainer.json', 'r') as f:
        dic = json.load(f)
    
    dic["name"] = args.competition_id
    
    mount_path = "source=${localWorkspaceFolder},target=/kaggle/input/" + args.ds_name + ",type=bind,consistency=cached"
    dic["workspaceMount"] = mount_path
    
    if args.device == "gpu":
        dic["runArgs"] = ["--init", "--gpus", "all", "--shm-size", "6gb"]
        if args.debug:
            raise ValueError("GPU mode does not support debug mode")
        else:
            dic["image"] = "gcr.io/kaggle-gpu-images/python:latest"
    else:
        dic["runArgs"] = ["--init"]
        if args.debug:
            dic["image"] = "mcr.microsoft.com/devcontainers/python:3.11-bullseye"
        else:
            dic["image"] = "gcr.io/kaggle-images/python:latest"
    
    dic["containerEnv"] = {
        "TZ": "Asia/Tokyo",
        "KAGGLE_DATASET_NAME": args.ds_name,
        "KAGGLE_COMPETITION_ID": args.competition_id,
    }

    with open(".devcontainer/devcontainer.json", "w") as f:
        json.dump(dic, f, indent=4)
    
    with open("source/env.py", "w") as f:
        f.write(f"dict = {dic['containerEnv']}")


if __name__ == '__main__':
    main()
