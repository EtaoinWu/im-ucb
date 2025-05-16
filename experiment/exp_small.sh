#!/bin/bash

seq 1 1024 | parallel -j60 --line-buffer ./build/bandit-im karate {} 0.1 0.01
