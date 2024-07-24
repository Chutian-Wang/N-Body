import numpy as np
import json
import warnings
from tqdm import tqdm

class Solver(object):
    def __init__(self, config_file = None, visualize = False) -> None:
        self._sim_length = None
        if visualize:
            from .Visual import Visualizer
            self.visualizer = Visualizer(self)
        else:
            self.visualizer = None
        if config_file is None:
            import os.path as path
            print("No config provided, using default demo data.")
            self.config_file = path.dirname(__file__) + "/default.json"
        else:
            self.config_file = config_file
        self._read_config()
        
    def __iter__(self):
        return self

    def __next__(self) -> "Solver":
        if self._sim_length is None:
            return self._update()
        elif self._sim_length > 0:
            self._sim_length -= 1
            return self._update()
        else:
            raise StopIteration()
        
    def __len__(self) -> int:
        return self._sim_length
    
    def __str__(self) -> str:
        return f"N-Body Solver object with {len(self.obj_pos)} objects."
    
    def _read_config(self) -> None:
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)
        self.dt = self.config["dt"]
        self.G  = self.config["G"]
        self.dtype = self.config["dtype"]
        self.objs = self.config["objs"]

    def _update(self) -> "Solver":
        mask = np.eye(len(self.obj_pos), dtype=bool)
        mask = ~mask
        dist = self.obj_pos[np.newaxis, :, :] - self.obj_pos[:, np.newaxis, :]
        mag = np.linalg.norm(dist, axis=-1)
        acc = self.obj_mass[:,np.newaxis] * dist / mag[:,:,np.newaxis]**3
        acc = np.nansum(acc,axis=1)
        self.obj_vel += self.dt * self.G * acc
        self.obj_pos += self.obj_vel * self.dt
        return self
    
    def build(self) -> None:
        self.obj_pos    = np.array([obj["pos"]
                            for obj in self.objs], dtype=self.dtype)
        self.obj_vel    = np.array([obj["vel"]
                            for obj in self.objs], dtype=self.dtype)
        self.obj_mass   = np.array([obj["mass"]
                            for obj in self.objs], dtype=self.dtype)
        
    def set_sim_length(self, length:any) -> None:
        if length is None:
            self._sim_length = None
        else:
            self._sim_length = int(length)
    
    def run(self, sim_length = None) -> None:
        if sim_length:
            self.set_sim_length(sim_length)
        if self._sim_length is None:
            print("Running simulation indefinitely.")
            print("If you'd like to set a limit, use set_sim_length() or prvode a length argument to run().")
            usr_input = input("Proceed? (y/n): ")
            while True:
                if usr_input.lower() == "n":
                    return
                elif usr_input.lower() == "y":
                    print("Press Ctrl+C to stop simulation.")
                    break
                else:
                    usr_input = input("Proceed? (y/n): ")

        with warnings.catch_warnings(action="ignore"):
            if self.visualizer:
                for _ in tqdm(self):
                    self.visualizer.update()
            else:
                for _ in tqdm(self):
                    pass