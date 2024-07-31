#!./.venv/bin/python3.11
from NBody import Solver
import argparse

parser = argparse.ArgumentParser(description='N-Body Simulation')
parser.add_argument('-c', '--config', metavar='<config_dir>', help='Specify the configuration directory')
parser.add_argument('-v', '--visualize', action='store_true', help='Visualize the simulation')
parser.add_argument('-p', '--precompute', action='store_true', help='Precompute the simulation by one max(1ms, 1frame), recommended if visualizing')
parser.add_argument('-t', '--time', metavar='<time>', type=float, help='The time in seconds for which to run the simulation')
parser.add_argument('-s', '--save', metavar='<save_dir>', help='Save the simulation result config file to <save_dir>')

args = parser.parse_args()

if args.time is None:
    print("No time specified, defaulting to 30 seconds")
    args.time = 30

if args.config is None:
    print("No config specified, using NBody module default config")

solver = Solver(args.config, args.visualize, args.precompute)
solver.run(args.time)
if args.save is not None:
    solver.save(args.save)