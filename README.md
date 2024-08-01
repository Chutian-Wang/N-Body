# N-Body Problem Numerical Solver

A primitive python N body solver that supports configurable visualization, video generation, incremental simulation, and various configurable built-in optimizations.

## Construction

This project uses the `numpy` package as its compute kernel and uses numerical integration to solve arbitrary N-Body problems. This project implements a `NBody.Solver` module in which a iterable `Solver` class is contained. The Solver object can load a simulation source and then solve the N-Body problem, optionally displaying or saving an animation of the simulation. It can also save the resulting data as a new simulation source for incremental simulation. For each iteration of the solver object, the solver updates its internal states (objects' position, velocity, etc.). If visualization is enabled, the solver will update its internal state to the time when the next frame should be captured. Otherwise, it advances itself by 1 millisecond in the simulation.

To achieve an iteration of the solver object, many smaller compute "ticks" are performed. For each tick with a defined small time interval, the gravitational acceleration for each object resulting from every other object is computed using classical mechanical assumptions and then integrated to get their respective velocities and positions in the next time frame. All computations are vectorized for maximum performance. Numerous optimizations such as frame precomputing are implemented to facilitate simulation and animation rendering.

A simulation source is provided with a configuration json file. A demonstration configuration file is included in the project directory. A demonstration application is also included, the usage of which can be accessed by running `./demo.py -h`

To view the usage of the Solver class, you may use the python `help` function.

```python
from NBody.Solver import Solver
help(Solver)
```

## Installation

Please `cd` into the project directory (N-Body) and run `install_packages.sh`. This script will install all necessary packages for this project and activate a virtual python environment at the current directory. Note that you need to manually deactivate this virtual environment by running `deactivate` if you wish to do so after leaving this project.

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
  -p, --precompute      Precompute the simulation by 1ms (or 1frame if visualization is enabled), recommended if visualizing
  -t <time>, --time <time>
                        The time in seconds for which to run the simulation
  -s <save_dir>, --save <save_dir>
                        Save the simulation result config file to <save_dir>

Example usage: ./demo.py -v -c demo_config.json -t 30 -s demo_result.json
```