#!/bin/sh
# postCreateCommand.sh

echo "START Install"

sudo chown -R vscode:vscode .

mkdir /kaggle/working

pip install pretty_errors
pip install mlflow

echo "FINISH Install"