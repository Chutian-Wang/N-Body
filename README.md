# N-Body Problem Numerical Solver

A primitive python N body solver that supports configurable visualization, video generation, incremental simulation, and various configurable built-in optimizations.

## Construction

The NBody module contains a `Solver` class, which can be instantiated from a config file. A developer may specify optimization options and visualization options upon instanciation or after the initialization of the solver object. Detailed documentation can be obtained with python function `help(Solver)`.

## Installation

Run `install_packages.sh` to create and activate a python virtual environment. This script will then install all the required packages specified in `requirements.txt` to the local python virtual environment.

## Demo

![demonstration gif](demo_video.gif)

A demonstration application is included for this project. Explore the options below.
usage: `./demo.py [-h] [-c <config_dir>] [-v] [-p] [-t <time>] [-s <save_dir>]`

```
options:
  -h, --help            show this help message and exit
  -c <config_dir>, --config <config_dir>
                        Specify the configuration directory
  -v, --visualize       Visualize the simulation
  -p, --precompute      Precompute the simulation by one max(1ms, 1frame), recommended if visualizing
  -t <time>, --time <time>
                        The time in seconds for which to run the simulation
  -s <save_dir>, --save <save_dir>
                        Save the simulation result config file to <save_dir>
```