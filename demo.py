from NBody import Solver

mysolver = Solver(visualize=False)
mysolver.build()
mysolver.run()
print(mysolver.obj_pos)