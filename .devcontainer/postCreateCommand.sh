#!/bin/sh
# postCreateCommand.sh

echo "START Install"

sudo chown -R vscode:vscode .

mkdir /kaggle/working

pip install pretty_errors
pip install mlflow

cd /kaggle/input/source
git clone https://github.com/jintonic3561/mlutil.git
cd /kaggle/input

echo "FINISH Install"