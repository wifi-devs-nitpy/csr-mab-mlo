#!/bin/bash

python tuning_2.py -a EGreedy -d egreedy.db
python tuning_2.py -a EGreedy -d flat_egreedy.db -f

python tuning_2.py -a Softmax -d softmax.db
python tuning_2.py -a Softmax -d flat_softmax.db -f

python tuning_2.py -a UCB -d ucb.db
python tuning_2.py -a UCB -d flat_ucb.db -f

python tuning_2.py -a NormalThompsonSampling -d ts.db
python tuning_2.py -a NormalThompsonSampling -d flat_ts.db -f
