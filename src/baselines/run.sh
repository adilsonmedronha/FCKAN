#!/bin/bash

DATASETS="Wafer HandOutlines Strawberry TwoPatterns DistalPhalanxOutlineCorrect"

for i in {1..10}; do
    python train.py --datasets $DATASETS --output_csv baselines_run_$i.csv
done
