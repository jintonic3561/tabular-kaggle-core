# Description
This repository provides a pipeline that can be used generically for Kaggle competitions with Tabular data.

# Get Started
To get started, execute .setting.py to set up the devcontainer.json.

```
python .setting.py --project_name my_project --ds_name my_dataset_name --device gpu 
```

Here is a summary of the command-line arguments for the .setting.py script:

| Argument | Description |
| --- | -------------------------- |
|--project_name | Specify the project name. |
|--ds_name | Specify the dataset title used for kaggle dataset. | 
|--device | Specify the device type. The expected values are "gpu" or "cpu". |
|--debug | If True, use a lightweight Python image as the docker image instead of a kaggle image. |

Note that this setup should be performed in your local Python environment.
Once the setup is complete, you can use the DevContainer feature of VSCode to build the container environment as usual.