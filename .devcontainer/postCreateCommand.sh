#!/bin/bash
# postCreateCommand.sh

echo "START Install"

sudo chown -R vscode:vscode .

mkdir /kaggle/working

# Install libraries that are not in Kaggle Image
pip install pretty_errors
pip install mlflow

# Install machine learning utility
git clone https://github.com/jintonic3561/mlutil.git /kaggle/input/$KAGGLE_DATASET_NAME/source/mlutil

# Download the competition dataset
chmod 600 /root/.kaggle/kaggle.json
kaggle competitions download -c $KAGGLE_COMPETITION_ID -p /kaggle/input
unzip /kaggle/input/$KAGGLE_COMPETITION_ID.zip -d /kaggle/input/$KAGGLE_COMPETITION_ID
rm /kaggle/input/$KAGGLE_COMPETITION_ID.zip

echo "FINISH Install"