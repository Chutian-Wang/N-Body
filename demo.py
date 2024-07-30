#!./.venv/bin/python3.11
from NBody import Solver

visulaize = input("Visualize? (y/n): ")
if visulaize.lower() == "y":
    mysolver = Solver(visualize=True)
else:
    mysolver = Solver()

mysolver.build()
mysolver.run(30)
print(mysolver.delta_E)
print(mysolver.time)