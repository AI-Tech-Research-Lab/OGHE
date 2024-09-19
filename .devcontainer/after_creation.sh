#/bin/bash

rm -r "/workspaces/PPDL4CancerClassification/openfhe-python"
cp -r "/openfhe-python" "/workspaces/PPDL4CancerClassification"
python3 -m pip install -U setuptools pip
python3 -m pip install -e openfhe-python 
python3 -m pip install numpy
python3 -m pip install torch
python3 -m pip install structlog
python3 -m pip install pickle
python3 -m pip install joblib
python3 -m pip install argparse
python3 -m pip install joblib
python3 -m pip install seaborn
python3 -m pip install scikit-learn
python3 -m pip install torchmetrics
python3 -m pip install matplotlib
python3 -m pip install lightning
