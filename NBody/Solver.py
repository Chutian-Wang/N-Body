import numpy as np
import json
import warnings
from tqdm import tqdm

class Solver(object):
    def __init__(self, config_file = None, visualize = False) -> None:
        self._sim_time = None
        self._time_left = float("inf")
        self.config_valid = False
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
        if self.config_valid:
            self.build()
        
    def __iter__(self) -> "Solver":
        return self

    def __next__(self) -> "Solver":
        if self._time_left > 0:
            return self._update()
        else:
            raise StopIteration()
    
    def __len__(self) -> int:
        return int(self._time_left * 1000)
    
    def __str__(self) -> str:
        return f"N-Body Solver object with {len(self.obj_pos)} objects."
    
    def _read_config(self) -> None:
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
            self.dt = self.config["dt"]
            self.G  = self.config["G"]
            self.dtype = self.config["dtype"]
            self.objs = self.config["objs"]
            self.config_valid = True
        except:
            print(f"Error loading config from {self.config_file}.")

    def _update(self) -> "Solver":
        # This needs to be changed if adaptive
        # time-stepping is to be implemented.
        dt = self.dt
        if self._time_left - self.dt < 0:
            dt = self._time_left
            self._time_left = 0
        else:
            self._time_left -= self.dt
        mask = np.eye(len(self.obj_pos), dtype=bool)
        mask = ~mask
        dist = self.obj_pos[np.newaxis, :, :] - self.obj_pos[:, np.newaxis, :]
        mag = np.linalg.norm(dist, axis=-1)
        acc = self.obj_mass[:,np.newaxis] * dist / mag[:,:,np.newaxis]**3
        acc = np.nansum(acc,axis=1)
        self.obj_vel += dt * self.G * acc
        self.obj_pos += self.obj_vel * dt
        return self
    
    def load_config(self, config_file: str) -> None:
        self.config_file = config_file
        self._read_config()
        if self.config_valid:
            self.build()
    
    def build(self) -> None:
        self.obj_pos    = np.array([obj["pos"]
                            for obj in self.objs], dtype=self.dtype)
        self.obj_vel    = np.array([obj["vel"]
                            for obj in self.objs], dtype=self.dtype)
        self.obj_mass   = np.array([obj["mass"]
                            for obj in self.objs], dtype=self.dtype)
        
    def set_sim_time(self, time: any) -> None:
        if time is None:
            self._sim_time = None
        else:
            self._sim_time = float(time)
    
    def run(self, time = None) -> None:
        if time:
            self.set_sim_time(time)

        self._time_left = self._sim_time

        if self._sim_time is None:
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
                for _ in self:
                    self.visualizer.update()
            else:
                progress_left = len(self)
                pbar = tqdm(total=progress_left, unit=" ms(simtime)", desc="Running simulation")
                for _ in self:
                    if len(self) < progress_left:
                        pbar.update(progress_left - len(self))
                        progress_left = len(self)
                pbar.close()