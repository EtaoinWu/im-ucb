#!/bin/bash

seq 1 50 | parallel -j50 --line-buffer ./build/bandit-im congress {} 10 0.01 --n_top 30 --lt
seq 1 50 | parallel -j50 --line-buffer ./build/bandit-im congress {} 0.003 0.05 --n_top 30 --lt --eval
