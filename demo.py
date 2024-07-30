#!./.venv/bin/python3.11
from NBody import Solver

mysolver = Solver(visualize=True)
mysolver.build()
mysolver.run(60)
print(mysolver.tsim_current)