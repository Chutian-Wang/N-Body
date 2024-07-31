#!./.venv/bin/python3.11
from NBody import Solver

visulaize = input("Visualize? (y/n): ")
if visulaize.lower() == "y":
    mysolver = Solver(visualize=True,precompute=True,config_file="demo_config.json")
else:
    mysolver = Solver()

mysolver.run(30)
