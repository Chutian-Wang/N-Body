import numpy as np
import json
import warnings
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from dataclasses import dataclass
from threading import Thread

COLORS = np.array(['r', 'g', 'b', 'c', 'm', 'y', 'k'])
MAX_FRAMES = 0xFFFFFFFF
DEFAULT_FPS = 30
DEFAULT_MS_PER_FRAME = 10

class Solver(object):
    """
    N-Body Solver class for simulating gravitational interactions between objects.
    """
    @dataclass
    class buffer:
        tsim_current: float
        tsim_start_ms: int
        tsim_current_ms: int
        obj_pos: np.ndarray
        obj_vel: np.ndarray

    @staticmethod
    def print_dict(d: dict, indent: int = 0) -> None:
        """
        Print a dictionary with indentation.

        Args:
            d (dict): The dictionary to print.
            indent (int, optional): The indentation level. Defaults to 0.
        """
        max_key_length = max(len(str(k)) for k in d.keys())
    
        for k, v in d.items():
            if isinstance(v, dict):
                print("  " * indent + str(k) + ":")
                Solver.print_dict(v, indent + 1)
            elif isinstance(v, list):
                print("  " * indent + str(k).ljust(max_key_length) + ": [")
                for item in v:
                    if isinstance(item, dict):
                        Solver.print_dict(item, indent + 1)
                    else:
                        print("  " * (indent + 1) + str(item))
                print("  " * indent + "]")
            else:
                # Align semicolons
                print("  " * indent + str(k).ljust(max_key_length) + ": " + str(v))

    def __init__(self, config_file=None, visualize=False, precompute = True) -> None:
        """
        Initialize the Solver object.

        Args:
            config_file (str, optional): Path to the configuration file. If not provided, default demo data will be used.
            visualize (bool, optional): Flag indicating whether to enable visualization. Defaults to False.
        """
        self.config_valid = False
        self.visualize = visualize
        self.precompute = precompute
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

        if self.visualize:
            print("Visualization properties:")
            Solver.print_dict(self.visuals, indent=1)
            print()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            scope = self.visuals.get("scope", [[-1, 1], [-1, 1], [-1, 1]])
            self.ax.set_xlim3d(scope[0])
            self.ax.set_ylim3d(scope[1])
            self.ax.set_zlim3d(scope[2])
            self.ax.set_title("3D Animated Scatter Plot")
            self.history = []
            self.colors = np.random.choice(COLORS, size=(self.obj_pos.shape[0]))
            self.lines = [self.ax.plot([], [], [], color = self.colors[i])[0] for i in range(self.obj_pos.shape[0])]
            self.scat = self.ax.scatter(self.obj_pos[:, 0], self.obj_pos[:, 1], self.obj_pos[:, 2], c=self.colors, marker='o')
            self.text_box = self.ax.text2D(-0.2, 0.8, '', transform=self.ax.transAxes)
            if self.visuals.get("disp_name", False):
                self.text_annotations = [self.ax.text(self.obj_pos[i, 0], self.obj_pos[i, 1], self.obj_pos[i, 2],
                                            self.objs[i]["name"], color=self.colors[i]) 
                                for i in range(self.obj_pos.shape[0])]
            else:
                self.text_annotations = None
            self.running = False

        if self.precompute:
            self.compute_thread = None
            self.__next__()

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

        # if precompute is enabled, self is immediately returned
        # and the next iteration is performed in a separate thread
        # along with plt's drawing thread (_update)
        if self.precompute:
            if self.tsim_ms and self.data_buf.tsim_current * 1000 - self.data_buf.tsim_start_ms >= self.tsim_ms:
                self.tsim_start_ms = self.tsim_current_ms
                raise StopIteration
            
            # in case the previous thread is still running
            if self.compute_thread:
                self.compute_thread.join()

            self.data_buf = self._gen_buffer()
            self.compute_thread = Thread(target=self._next_ms)
            self.compute_thread.start()
            
        else:
            if self.tsim_ms and self.tsim_current * 1000 - self.tsim_start_ms >= self.tsim_ms:
                self.tsim_start_ms = self.tsim_current_ms
                raise StopIteration
            
            if self.visualize:
                while self.tsim_current*1000 < self.tsim_current_ms + self.visuals.get("ms_per_frame", DEFAULT_MS_PER_FRAME):
                    self._tick()
                self.tsim_current_ms += self.visuals.get("ms_per_frame", DEFAULT_MS_PER_FRAME)
            else:
                while self.tsim_current*1000 < self.tsim_current_ms + 1:
                    self._tick()
                self.tsim_current_ms += 1
        
        return self
        
    def __len__(self) -> int:
        """
        Get simulation length in ms.

        Returns:
            int: simulation length in ms.
        """
        if self.tsim_ms is None and self.visualize:
            return MAX_FRAMES
        return self.tsim_ms

    def __str__(self) -> str:
        """
        Get a string representation of the Solver object.

        Returns:
            str: A string describing the Solver object.
        """
        return f"N-Body Solver object with {len(self.obj_pos)} objects."
    
    def _next_ms(self) -> None:
        
        if self.visualize:
            while self.tsim_current*1000 < self.tsim_current_ms + self.visuals.get("ms_per_frame", DEFAULT_MS_PER_FRAME):
                self._tick()
            self.tsim_current_ms += self.visuals.get("ms_per_frame", DEFAULT_MS_PER_FRAME)
        else:
            while self.tsim_current*1000 < self.tsim_current_ms + 1:
                self._tick()
            self.tsim_current_ms += 1

    def _gen_buffer(self) -> None:
        return self.buffer(self.tsim_current, self.tsim_start_ms, self.tsim_current_ms, self.obj_pos.copy(), self.obj_vel.copy())
    
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
            self.desc = self.config["desc"]
            if self.visualize:
                self.visuals = self.config["visuals"]
            self.config_valid = True
        except:
            print(f"Error loading config from {self.config_file}.")

    def _tick(self) -> "Solver":
        """
        Perform an update step in the simulation.

        Returns:
            Solver: The Solver object itself.
        """
        # this is to suppress the RuntimeWarning: invalid value encountered in divide
        # this is due to object self referencing calculating acceleration
        # this line alone costs 1/16 of the total runtime when problem sizes are small
        # but it's not a big deal as problem sizes get larger
        warnings.filterwarnings('ignore', r'invalid value*')
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
    
    # FuncAnimation provides self here,
    # so we need to declear it as a static method
    @staticmethod
    def _update(self) -> None:
        if self.precompute:
            data = self.data_buf
        else:
            data = self

        if data.tsim_current_ms % self.visuals.get("ms_per_frame", DEFAULT_MS_PER_FRAME) == 0:
            if data.tsim_current_ms == self.tsim_ms:
                self.text_box.set_text(fr"simulation time: {data.tsim_current:.2f} s" + \
                                '\n' + fr"$\Delta E = {self.delta_E:.2f} J$" + \
                                f"({self.delta_E / self.E:.1f}%)\n" + \
                                "Simulation complete.")
            else:
                self.text_box.set_text(fr"simulation time: {data.tsim_current:.2f} s" + \
                                    '\n' + fr"$\Delta E = {self.delta_E:.2f} J$" + \
                                    f"({self.delta_E / self.E:.1f}%)\n" + \
                                    f"{self.visuals.get('ms_per_frame', DEFAULT_MS_PER_FRAME)} ms per frame @ {self.visuals.get('fps', DEFAULT_FPS)} fps")
            
            if self.text_annotations:
                for i, text in enumerate(self.text_annotations):
                    text.set_position((data.obj_pos[i, 0], data.obj_pos[i, 1]))
                    text.set_3d_properties(data.obj_pos[i, 2], zdir = (1,1,0))
                    text.set_text(self.objs[i]["name"])

            self.scat._offsets3d = (data.obj_pos[:, 0], data.obj_pos[:, 1], data.obj_pos[:, 2])
            self.history.append(data.obj_pos.copy())
            
            # Limit history length for trace
            trace_length = self.visuals.get("trace_length", 20)
            if len(self.history) > trace_length:
                self.history.pop(0)
                
            # Update trace lines
            for i in range(data.obj_pos.shape[0]):
                trace_data = np.array(self.history)[:, i, :]
                self.lines[i].set_data(trace_data[:, 0], trace_data[:, 1])
                self.lines[i].set_3d_properties(trace_data[:, 2])
    
        return self.scat, *self.lines, self.text_box
    
    def _build(self) -> None:
        """
        Build the initial state of the simulation based on the loaded configuration.
        """
        self.obj_pos = np.array([obj["pos"]
                                 for obj in self.objs], dtype=self.dtype)
        self.obj_vel = np.array([obj["vel"]
                                 for obj in self.objs], dtype=self.dtype)
        self.obj_mass = np.array([obj["mass"]
                                  for obj in self.objs], dtype=self.dtype)
    
    def load_config(self, config_file: str) -> None:
        """
        Load a new configuration file.

        Args:
            config_file (str): Path to the new configuration file.
        """
        self.config_file = config_file
        self._read_config()
        if self.config_valid:
            self._build()
        else:
            raise ValueError("Invalid configuration file.")

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
        if self.precompute:
            return self.data_buf.tsim_current
        return self.tsim_current
    
    @property
    def delta_E(self) -> float:
        if self.precompute:
            return self.E - np.sum(self.obj_mass * np.linalg.norm(self.data_buf.obj_vel, axis=1)**2 / 2)
        return self.E - np.sum(self.obj_mass * np.linalg.norm(self.obj_vel, axis=1)**2 / 2)

    def run(self, time=None) -> None:
        """
        Run the simulation.

        Args:
            time (any, optional): The total simulation time. Set to None for indefinite simulation.
        """
        self.set_sim_time(time)
        print(self.desc)

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

        if (not hasattr(self, "ani")) and self.visualize:
            self.ani = FuncAnimation(self.fig, self._update, frames=self, interval=self.visuals.get("fps", DEFAULT_FPS) , blit=False)

        try:
            if self.visualize:
                self.running = True
                plt.show()
            else:
                for _ in tqdm(self, unit="ms"):
                    pass
                
        except KeyboardInterrupt:
            print("Simulation stopped.")

    def save(self, filename: str) -> None:
        """
        Save the current state of the simulation to a new configuration file.
        This is for incremental saving of the simulation result.
        Args:
            filename (str): The path to the file where the configuration will be saved.
        """
        config_data = {
            "dt": self.dt,
            "G": self.G,
            "dtype": self.dtype,
            "objs": [
                {
                    "pos": self.obj_pos[i].tolist(),
                    "vel": self.obj_vel[i].tolist(),
                    "mass": self.obj_mass[i],
                    "name": self.objs[i]["name"]
                } for i in range(len(self.objs))
            ],
            "desc": self.desc
        }
        if self.config.get("visuals", None):
            config_data["visuals"] = self.config["visuals"]
        
        with open(filename, 'w') as f:
            json.dump(config_data, f, indent=4)
        print(f"Simulation result configuration saved to {filename}")
