# Description
This repository provides a pipeline that can be used generically for Kaggle competitions with Tabular data.

# Get Started
To get started, execute .setting.py to set up the devcontainer.json.

```
python .setting.py --competition_id join_competition_id --ds_name your_dataset_name --device gpu 
```

Here is a summary of the command-line arguments for the .setting.py script:

| Argument | Description |
| ------ | -------------------------- |
|--competition_id | Specify the project name. |
|--ds_name | Specify the dataset title used for kaggle dataset. | 
|--device | Specify the device type. The expected values are "gpu" or "cpu". |
|--debug | If True, use a lightweight Python image as the docker image instead of a kaggle image. |

Note that this setup should be performed in your local Python environment.
Once the setup is complete, you can use the DevContainer feature of VSCode to build the container environment as usual.

### Waring
Note that git can automatically change the newline code. In particular, `.devcontainer/postCreateCommand.sh` will not work properly unless the newline code is LF.

# Clone as Your Private Repository
0. Create an empty private Git repository
1. Clone this repository
```
git clone https://github.com/jintonic3561/tabular-kaggle-core.git your-repository-name -b main
```
2. Configure remote url
```
cd your-repository-name
git remote add abs https://github.com/jintonic3561/tabular-kaggle-core.git
git remote set-url origin your-private-git-repository.git
git remote -v
```
3. Push to your private repository
```
git push origin main
```
4. Pull updates from this repository (If needed)
```
git pull abs main
```