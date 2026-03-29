#!/bin/bash

(python tuning_2.py -a EGreedy -d egreedy.db) &

(python tuning_2.py -a Softmax -d softmax.db) &

(python tuning_2.py -a UCB -d ucb.db) &

(python tuning_2.py -a NormalThompsonSampling -d ts.db)
