#!./.venv/bin/python3
from NBody.Solver import Solver
import argparse

parser = argparse.ArgumentParser(description='N-Body Simulation')
parser.add_argument('-c', '--config', metavar='<config_dir>', help='Specify the configuration directory')
parser.add_argument('-v', '--visualize', action='store_true', help='Visualize the simulation')
parser.add_argument('-p', '--precompute', action='store_true', help='Precompute the simulation by 1ms (or 1frame if visualization is enabled), recommended if visualizing')
parser.add_argument('-t', '--time', metavar='<time>', type=float, help='The time in seconds for which to run the simulation')
parser.add_argument('-s', '--save', metavar='<save_dir>', help='Save the simulation result config file to <save_dir>')

parser.epilog = "Example usage: ./demo.py -v -c demo_config.json -t 30 -s demo_result.json"

args = parser.parse_args()

if args.time is None:
    print("No time specified, defaulting to 30 seconds")
    args.time = 30

if args.config is None:
    print("No config specified, using NBody module default config")

solver = Solver(args.config, args.visualize, args.precompute)
# if you want to save the animation instead of viewing it,
# you can specify the path to save the animation
# solver.run(args.time,animation_path="demo_video.gif")
solver.run(args.time)
# save the simulation result to a config file
if args.save is not None:
    solver.save(args.save)