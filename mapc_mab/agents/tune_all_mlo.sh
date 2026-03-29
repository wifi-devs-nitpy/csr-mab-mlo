#!/bin/bash

python tuning_2_jax.py -a EGreedy -d egreedy.db

python tuning_2_jax.py -a Softmax -d softmax.db

python tuning_2_jax.py -a UCB -d ucb.db

python tuning_2_jax.py -a NormalThompsonSampling -d ts.db
