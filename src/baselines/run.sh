#!/bin/bash

DATASETS="Wafer HandOutlines Strawberry TwoPatterns DistalPhalanxOutlineCorrect Chinatown ItalyPowerDemand ECG200 ArrowHead CricketX CricketY CricketZ Beef"
RUNS=10
for i in $(seq 1 $RUNS); do
    python train.py --seed $i --datasets $DATASETS --output_csv baselines_run_$i.csv
done
