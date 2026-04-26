# Introduction
This repository contains all the code used to train and test different reinforcement algorithms as well as all the hyper-parameters selected for each run and the results data. 

## Results Data and Hyper-Parameters
The hyper-parameters for each maze/map size, both static and dynamic can be found in the ”Results” folder. The folder contains sub-folders named ”Static Mazes” and ”Dynamic Mazes” for each experiment type. Inside both of them are folders named after the maze sizes and then inside each are folders named after each tested algorithm. 

Then, inside each of those algorithms are sub-folders for each trial runs and a ”parameters.txt” file containing the exact hyper-parameters used to run those trials. Finally, each trial run folder contains the data collected for each run. This data is the memory usage, training time, training graph showing average rewards per training episode, testing graph showing average rewards and steps per training episode and the cvs files containing the data used to generate both graphs. 

It is important to note that the memory usage file contains memory samples taken every 0.01 seconds as well as the peak memory usage of the training run and the average memory usage of the training run at the end of the file.

## Code
The code run on either on OpenAI's Gymnasium "cliff walking" or "frozen lake" environments.

There are 7 main variations. Perceptive PPO-CNN is in the same folder as perceptive PPO (./perceptive-proximal-policy-optimization).

List of variations:
- Q-Leaning
- SARSA
- DQL/DQN
- PPO
- Perceptive DQL/DQN
- Perceptive PPO
- Perceptive PPO with CNN networks.

The Perceptive implementations get fed tile information from the map as well as agent position and relative goal position from the agent.
The perceptive implementations seems to be rather unique for "frozen lake". They were inspired by MiniGrid's "FlatObsWrapper" function.
See: https://minigrid.farama.org/api/wrapper/


In other to run them all one after the other use "run.sh". This file is only works on Linux.

run.sh arguments:
run [mode] [size] [perceptive] [cnn]

Values:
mode - test, train (determines what mode to run on, be it to train a model or test it.)
size - 4, 8, 12, 16, 20 (determines what map/maze size to run for frozen lake, doesn't work on cliff walking)
perceptive yes, no (whether to also run the perceptive DQL and PPO algorithms)
cnn - yes, no (whether to also run the perceptive PPO-CNN algorithm. Warning: very heavy.)

This code is freely available for anyone to use and modify.
Credits where certain code sections were inspired from inside each file.
