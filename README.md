# Influence Maximization with Confidence Bound optimizations

This is an implementation of the greedy and CELF algorithms for influence maximization.  
We also include an optimization for both of them using confidence bounds.

This is a course project for [CSCI 673](http://www.david-kempe.com/CS673/index.html).

## Modern C++ API

- Core symbols are provided in namespace `im`.
- A module-friendly umbrella header is available at `include/im/all.hpp`.

## Usage

To build, use `cmake`:

```bash
mkdir build
cd build
cmake ..
make
```

To run, use `./bandit-im`:

```bash
./bandit-im <dataset> <seed> <epsilon> <delta>
```

Look at `experiment/exp_*.sh` for examples. The visualization is done in `experiment/visualize.ipynb`.

## Unit tests

To run unit tests, use `ctest`:

```bash
ctest
```
