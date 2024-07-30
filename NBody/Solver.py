import numpy as np
import json
import warnings
from .Visual import Visualizer
from tqdm import tqdm

import json
import numpy as np
import warnings

class Solver(object):
    """
    N-Body Solver class for simulating gravitational interactions between objects.
    """

    def __init__(self, config_file=None, visualize=False) -> None:
        """
        Initialize the Solver object.

        Args:
            config_file (str, optional): Path to the configuration file. If not provided, default demo data will be used.
            visualize (bool, optional): Flag indicating whether to enable visualization. Defaults to False.
        """
        self.config_valid = False
        self.visualize = visualize
        self.tsim_current = 0
        self.tsim_ms = 0

        self.tsim_start_ms = 0
        self.tsim_current_ms = 0

        if config_file is None:
            import os.path as path
            print("No config provided, using default demo data.")
            self.config_file = path.dirname(__file__) + "/default.json"
        else:
            self.config_file = config_file

        self.load_config(self.config_file)
        self.E = np.sum(self.obj_mass * np.linalg.norm(self.obj_vel, axis=1)**2 / 2)

    def __iter__(self) -> "Solver":
        """
        Enable iteration over the Solver object.

        Returns:
            Solver: The Solver object itself.
        """
        return self

    def __next__(self) -> "Solver":
        """
        Perform the next iteration of the simulation.

        Returns:
            Solver: The Solver object itself.

        Raises:
            StopIteration: If the simulation has reached its end.
        """

        if self.tsim_current * 1000 - self.tsim_start_ms >= self.tsim_ms:
            self.tsim_start_ms = self.tsim_current_ms
            raise StopIteration
        
        if self.visualize:
            while self.tsim_current*1000 < self.tsim_current_ms + self.ms_per_frame:
                self._update()
            self.tsim_current_ms += self.ms_per_frame
        else:
            while self.tsim_current*1000 < self.tsim_current_ms + 1:
                self._update()
            self.tsim_current_ms += 1
        
        return self
        
    def __len__(self) -> int:
        """
        Get simulation length in ms.

        Returns:
            int: simulation length in ms.
        """
        return self.tsim_ms

    def __str__(self) -> str:
        """
        Get a string representation of the Solver object.

        Returns:
            str: A string describing the Solver object.
        """
        return f"N-Body Solver object with {len(self.obj_pos)} objects."

    def _read_config(self) -> None:
        """
        Read the configuration from the specified file.
        """
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
            self.dt = self.config["dt"]
            self.G = self.config["G"]
            self.dtype = self.config["dtype"]
            self.objs = self.config["objs"]
            if self.visualize:
                self.scope = np.array(self.config["scope"])
                self.ms_per_frame = self.config["ms_per_frame"]
            self.config_valid = True
        except:
            print(f"Error loading config from {self.config_file}.")

    def _update(self) -> "Solver":
        """
        Perform an update step in the simulation.

        Returns:
            Solver: The Solver object itself.
        """
        dt = self.dt
        self.tsim_current += dt
        mask = np.eye(len(self.obj_pos), dtype=bool)
        mask = ~mask
        dist = self.obj_pos[np.newaxis, :, :] - self.obj_pos[:, np.newaxis, :]
        mag = np.linalg.norm(dist, axis=-1)
        acc = self.obj_mass[:, np.newaxis] * dist / mag[:, :, np.newaxis] ** 3
        acc = np.nansum(acc, axis=1)
        self.obj_vel += dt * self.G * acc
        self.obj_pos += self.obj_vel * dt
        return self

    def load_config(self, config_file: str) -> None:
        """
        Load a new configuration file.

        Args:
            config_file (str): Path to the new configuration file.
        """
        self.config_file = config_file
        self._read_config()
        if self.config_valid:
            self.build()

    def build(self) -> None:
        """
        Build the initial state of the simulation based on the loaded configuration.
        """
        self.obj_pos = np.array([obj["pos"]
                                 for obj in self.objs], dtype=self.dtype)
        self.obj_vel = np.array([obj["vel"]
                                 for obj in self.objs], dtype=self.dtype)
        self.obj_mass = np.array([obj["mass"]
                                  for obj in self.objs], dtype=self.dtype)

    def set_sim_time(self, time: any) -> None:
        """
        Set the total simulation time in ms.

        Args:
            time (any): The total simulation time. Set to None for indefinite simulation.
        """
        if time is None:
            self.tsim_ms = None
        else:
            self.tsim_ms = int(time * 1000)

    @property
    def time(self) -> float:
        """
        Get the current simulation time in seconds.

        Returns:
            float: The current simulation time in seconds.
        """
        return self.tsim_current
    
    @property
    def delta_E(self) -> float:
        return self.E - np.sum(self.obj_mass * np.linalg.norm(self.obj_vel, axis=1)**2 / 2)

    def run(self, time=None) -> None:
        """
        Run the simulation.

        Args:
            time (any, optional): The total simulation time. Set to None for indefinite simulation.
        """
        self.set_sim_time(time)

        if self.tsim_ms is None:
            print("Running simulation indefinitely.")
            print("If you'd like to set a limit, use set_sim_length() or provide a length argument to run().")
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
            try:
                if self.visualize:
                    self.visualizer = Visualizer(self)
                else:
                    for _ in tqdm(self, unit="ms"):
                        pass
                    
            except KeyboardInterrupt:
                print("Simulation stopped.")

        self._time_left = float("inf")