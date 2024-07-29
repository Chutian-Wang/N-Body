#!./.venv/bin/python3.11
from NBody import Solver

mysolver = Solver(visualize=False)
mysolver.build()
mysolver.run(10000)
print(mysolver.obj_pos)