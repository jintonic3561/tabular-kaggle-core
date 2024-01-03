#!/bin/sh
# postCreateCommand.sh

echo "START Install"

sudo chown -R vscode:vscode .

mkdir /kaggle/working

# Install libraries that are not in Kaggle Image
pip install pretty_errors
pip install mlflow

# Install machine learning utility
cd /kaggle/input/${KAGGLE_DATASET_NAME}/source
git clone https://github.com/jintonic3561/mlutil.git

# Download the competition dataset
cd /kaggle/input
kaggle competitions download -c ${KAGGLE_COMPETITION_ID}

cd /kaggle
echo "FINISH Install"