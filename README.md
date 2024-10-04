## Installation
We suggest creating a python virtual environment with Python3.10, then after activating the environment type:
```
make install
```
to install all the requirements.

## Datasets availability
All datasets are publicly available.
Datasets are generated (eventually downloaded) on-the-fly and then cached when running a training that requires a dataset. In order to generate a dataset manually you can run the relative python script in the `daset_scripts` folder. The dataset will be stored in the `data` folder.

## Training the model
```
python main-train.py config.json
```
trains a model and stores the results in the `artifacts/models` folder.