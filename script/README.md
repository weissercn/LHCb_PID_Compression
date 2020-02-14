# LHCb PID compression script

## Setup

```
pip install -r requirements.txt
```


## Run

```
compress.py:
KERAS_BACKEND=tensorflow python compress.py kaon input.csv output.csv

generate.py:
KERAS_BACKEND=tensorflow python generate.py kaon input.csv output.csv
```
