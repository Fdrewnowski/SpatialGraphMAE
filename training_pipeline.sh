#!/bin/sh

python ./bikeguessr_download_cycleway.py --all
python ./bikeguessr_transform.py --all
python ./bikeguessr_train.py --use_cfg
python ./bikeguessr_predict.py --save
