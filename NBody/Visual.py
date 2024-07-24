import matplotlib.pyplot as plt
import numpy as np
from .Solver import Solver

class Visualizer(object):
    def __init__(self, solver) -> None:
        self.solver = solver

    def set_scope(self, scope) -> None:
        raise NotImplementedError("Visualizer.set_scope not implemented.")
    
    def update(self) -> None:
        raise NotImplementedError("Visualizer.update not implemented.")