# Sentiment analysis for Twitter
Short intro. We provide various preprocessing techniques as well as 4 models.

Please refer to XY paper for a description of the methods used, refer to Structure for a high level overview over the repository, refer to Prerequisits for installation instructions.

## Installation

For smooth running of the preprocessing and models we use pipenv library to manage virtual environments and dependencies on different devices. You can install via

1. Install `pipenv` with
```
pip install pipenv
```
2. Spin up a virtual environment with:
```
pipenv shell
```
3. Install dependencies (`--dev` option needed for some of the preprocessing steps)
```
pipenv install
```


## Structure
run_predict.py
run_train.py
etc.

## Prerequisites

#### Pip dependencies
requirements_dev.txt
requirements_prod.txt

#### Preprocessed data sets
For convinience, we provide links to the already processed data sets as downloadable zip folders.
etc.

#### Frameworks used
CUDA pytorch and tf version.
