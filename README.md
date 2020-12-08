# Othello

A set of tools to play Othello using neural networks and MCTS, strongly inspired by AlphaZero.

An Nvidia GPU is required.

### Running

This project consists of several subprograms. These are:

##### Analysis - `bin/analyze`
Analyse a given position and print various information about the search, including the principal variation (PV) and an estimation of the expected winner.

##### Self Play - `bin/self_play`
Play games against itself, to generate data to improve the neural network.

##### Augment - `bin/augment`
Convert data from self play format to one suitable for training. This additionally uses rotation and reflection of the board to increase the amount of data available.

##### Training - `train.py`
Use data generated by the previous steps to improve the current neural network.

##### Testing - `bin/testing`
Run a tournament between several neural networks so as to make comparisons.

### Building

A C++17 compiler is required, along CUDA and cuDNN.

Running `make` will build all of the above tools.

`make clean` can be used to remove all generated files.

### Example Analysis

P refers to the prior probability of a move as calculated by the neural network.

Nc / Np is the proportion of MCTS simulations which explored the given move. The node scoring highest according to this metric is considered the best.

Q is the expected value of that move.

![Analysis](https://maf27.host.cs.st-andrews.ac.uk/Othello/ExampleAnalysis.png)
