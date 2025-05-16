#!/bin/bash

seq 1 50 | parallel -j50 --line-buffer ./build/bandit-im congress {} 25 0.01 --n_top 30
