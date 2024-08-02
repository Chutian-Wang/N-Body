import json
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
from threading import Thread
from tqdm import tqdm
from dataclasses import dataclass
import warnings
import time

import matplotlib.pyplot as plt

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

    def __init__(self, config:any=None, visualize=False, precompute=True, title='') -> None:
        """
        Initialize the Solver object.

        Args:
            config (any, optional): Configuration data/path for the solver. If not provided, default demo data will be used.
            visualize (bool, optional): Flag indicating whether to enable visualization. Defaults to False.
            precompute (bool, optional): Flag indicating whether to enable precomputation by 1 time frame. Defaults to True.
            title (str, optional): Title for the visualization. Defaults to an empty string.
        """
        self.perf_c = time.perf_counter()
        self.export = False
        self.done = False
        self.visualize = visualize
        self.precompute = precompute
        self.title = title
        self.tsim_current = 0
        self.tsim_ms = 0

        self.tsim_start_ms = 0
        self.tsim_current_ms = 0

        self.load(config)
        
        self.E = np.sum(self.obj_mass * np.linalg.norm(self.obj_vel, axis=1)**2 / 2)

        if self.visualize:
            print("Visualization properties:")
            Solver.print_dict(self.visuals, indent=1)
            print()

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
        if self.tsim_ms and len(self) == 0:
            if self.precompute:
                self.tsim_start_ms = self.data_buf.tsim_current_ms
                self.data_buf.tsim_start_ms = self.tsim_current_ms
            else:
                self.tsim_start_ms = self.tsim_current_ms
            self.done = True
            raise StopIteration
        
        # if precompute is enabled, self is immediately returned
        # and the next iteration is performed in a separate thread
        # along with plt's drawing thread (_update)
        if self.precompute:
            # in case the previous thread is still running
            if self.compute_thread:
                self.compute_thread.join()

            self.data_buf = self._gen_buffer()
            self.compute_thread = Thread(target=self._next)
            self.compute_thread.start()
            
        else:
            
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
        Get simulation length in ms or frames (if visualization is enabled).

        Returns:
            int: simulation length in ms or frames (if visualization is enabled).
        """
        if self.tsim_ms is None and self.visualize:
            return MAX_FRAMES
        if self.visualize:
            data = self.data_buf if self.precompute else self
            return int(np.ceil((self.tsim_ms - data.tsim_current_ms + data.tsim_start_ms) / self.visuals.get("ms_per_frame", DEFAULT_MS_PER_FRAME)))
        else:
            data = self.data_buf if self.precompute else self
            return self.tsim_ms - data.tsim_current_ms + data.tsim_start_ms

    def __str__(self) -> str:
        """
        Get a string representation of the Solver object.

        Returns:
            str: A string describing the Solver object.
        """
        return f"N-Body Solver object with {len(self.obj_pos)} objects."
    
    def _next(self) -> None:
        
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
    
    def _read_config(self, file_path) -> None:
        try:
            with open(file_path, 'r') as f:
                self.config = json.load(f)
        except:
            raise ValueError(f"Error loading config from {self.config_file}.")

    def _tick(self) -> "Solver":
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
    
    def _make_fig(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        scope = self.visuals.get("scope", [[-1, 1], [-1, 1], [-1, 1]])
        self.ax.set_xlim3d(scope[0])
        self.ax.set_ylim3d(scope[1])
        self.ax.set_zlim3d(scope[2])
        self.ax.set_title(self.title)
        self.history = []
        self.colors = np.random.choice(COLORS, size=(self.obj_pos.shape[0]))
        self.lines = [self.ax.plot([], [], [], color=self.colors[i])[0] for i in range(self.obj_pos.shape[0])]
        self.scat = self.ax.scatter(self.obj_pos[:, 0], self.obj_pos[:, 1], self.obj_pos[:, 2], c=self.colors, marker='o')
        self.text_box = self.ax.text2D(-0.2, 0.8, '', transform=self.ax.transAxes)
        if self.visuals.get("disp_name", False):
            self.text_annotations = [self.ax.text(self.obj_pos[i, 0], self.obj_pos[i, 1], self.obj_pos[i, 2],
                                                    self.config["objs"][i]["name"], color=self.colors[i])
                                        for i in range(self.obj_pos.shape[0])]
        else:
            self.text_annotations = None
    
    # FuncAnimation provides self here,
    # so we need to declear it as a static method
    @staticmethod
    def _update(self) -> None:
        now = time.perf_counter()
        true_fps = 1 / (now - self.perf_c)
        self.perf_c = now
        if self.precompute:
            data = self.data_buf
        else:
            data = self

        if data.tsim_current_ms - data.tsim_start_ms == self.tsim_ms:
            self.text_box.set_text(fr"simulation time: {data.tsim_current:.2f} s" + \
                            '\n' + fr"$\Delta E = {self.delta_E:.2f} J$" + \
                            f"({self.delta_E / self.E:.1f}%)\n" + \
                            "Simulation complete.")
        else:
            fps = self.visuals.get("fps", DEFAULT_FPS) if self.export else true_fps
            self.text_box.set_text(fr"simulation time: {data.tsim_current:.2f} s" + \
                                '\n' + fr"$\Delta E = {self.delta_E:.2f} J$" + \
                                f"({self.delta_E / self.E:.1f}%)\n" + \
                                f"{self.visuals.get('ms_per_frame', DEFAULT_MS_PER_FRAME)} ms per frame @ {fps:.1f} fps")
        
        if self.text_annotations:
            for i, text in enumerate(self.text_annotations):
                text.set_position((data.obj_pos[i, 0], data.obj_pos[i, 1]))
                text.set_3d_properties(data.obj_pos[i, 2], zdir = (1,1,0))

        self.scat._offsets3d = (data.obj_pos[:, 0], data.obj_pos[:, 1], data.obj_pos[:, 2])
        self.history.append(data.obj_pos.copy())
        
        # Limit history length for trace
        trace_length = self.visuals.get("trace_length", 20)
        if trace_length:
            if len(self.history) > trace_length:
                self.history.pop(0)
            
            # Update trace lines
            for i in range(data.obj_pos.shape[0]):
                trace_data = np.array(self.history)[:, i, :]
                self.lines[i].set_data(trace_data[:, 0], trace_data[:, 1])
                self.lines[i].set_3d_properties(trace_data[:, 2])
    
        return self.scat, *self.lines, self.text_box
    
    def _build(self) -> None:

        try:
            self.dt = self.config["dt"]
            self.G = self.config["G"]
            if self.visualize:
                self.visuals = self.config["visuals"]

                if (fps := self.visuals.get("fps", None)) is None:
                    warnings.warn("fps not specified, using 30.")
                elif not isinstance(fps, int) or fps <= 0:
                    raise ValueError("fps must be a positive integer.")
                
                if (ms_per_frame := self.visuals.get("ms_per_frame", None)) is None:
                    warnings.warn("ms_per_frame not specified, using 10.")
                elif not isinstance(ms_per_frame, int) or ms_per_frame <= 0:
                    raise ValueError("ms_per_frame must be a positive integer.")
                
                if (trace_length := self.visuals.get("trace_length", None)) is None:
                    warnings.warn("trace_length not specified, using 20.")
                elif not isinstance(trace_length, int) or trace_length < 0:
                    raise ValueError("trace_length must be a non-negative integer.")
                
                if self.visuals.get("scope", None) is None:
                    raise ValueError("scope not specified.")
            
            if self.config.get("dtype", None) is None:
                warnings.warn("dtype not specified, using float32.")

            if self.config.get("objs", None) is None:
                raise ValueError("No objects specified.")

            self.obj_pos = np.array([obj["pos"]
                                    for obj in self.config["objs"]], dtype=self.config.get("dtype", np.float32))
            self.obj_vel = np.array([obj["vel"]
                                    for obj in self.config["objs"]], dtype=self.config.get("dtype", np.float32))
            self.obj_mass = np.array([obj["mass"]
                                    for obj in self.config["objs"]], dtype=self.config.get("dtype", np.float32))
        except:
            raise ValueError("Invalid configuration.")
    
    def load(self, config: any) -> None:
        """
        Load a configuration.

        Args:
            config (any): The configuration to load. It can be a dictionary, a path to a configuration file, or None.

        Raises:
            ValueError: If the configuration is any of the above.

        Returns:
            None
        """
        if config is None:
            import os.path as path
            file_path = path.dirname(__file__) + "/default.json"
            self._read_config(file_path)
        elif isinstance(config, dict):
            self.config = config
            self._build()
        elif isinstance(config, str):
            self._read_config(config)
        else:
            raise ValueError("Configurations must be a dictionary or a path to a configuration file.")

        self._build()

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
        """
        Calculate the energy difference from the beginning of the simulation to the current state.
        This is useful if simulation confidence is of interest.

        Returns:
            float: The energy difference.
        """
        if self.precompute:
            return self.E - np.sum(self.obj_mass * np.linalg.norm(self.data_buf.obj_vel, axis=1)**2 / 2)
        return self.E - np.sum(self.obj_mass * np.linalg.norm(self.obj_vel, axis=1)**2 / 2)

    def run(self, time=None, animation_path=None) -> None:
            """
            Run the simulation.

            Args:
                time (any, optional): The total simulation time. Set to None for indefinite simulation.
                animation_path (str, optional): The file path to save the animation. If not provided, the animation will be displayed on the screen.

            Returns:
                None

            Raises:
                KeyboardInterrupt: If the simulation is stopped by pressing Ctrl+C.

            Notes:
                - If `time` is set to None, the simulation will run indefinitely until manually stopped.
                - If `animation_path` is provided, the animation will be saved as a file using the specified path.
                - If `animation_path` is not provided, the animation will be displayed on the screen.
            """
            self.done = False
            if time: self.set_sim_time(time)
            if self.precompute:
                if self.data_buf.tsim_current_ms == self.tsim_ms:
                    warnings.warn("Simulation already ran, running more may cause matplotlib to have unexpected behavior.")
                    #return
            else:
                if self.tsim_current_ms == self.tsim_ms:
                    warnings.warn("Simulation already ran, running more may cause matplotlib to have unexpected behavior.")
                    #return

            print(self.config.get("desc", "No description."))

            if self.tsim_ms is None:
                warnings.warn("Running simulation indefinitely.")
                warnings.warn("animation_path is ignored.")
                print("If you'd like to set a limit, use set_sim_length() or provide a length argument to run().")
                usr_input = input("Proceed? (y/n): ")
                while True:
                    if usr_input.lower() == "n":
                        return
                    elif usr_input.lower() == "y":
                        animation_path = None
                        print("Press Ctrl+C to stop simulation.")
                        break
                    else:
                        usr_input = input("Proceed? (y/n): ")

            if self.visualize:
                self._make_fig()
                self.ani = FuncAnimation(self.fig,
                            self._update,
                            frames=self,
                            interval=self.visuals.get("fps", DEFAULT_FPS),
                            blit=False,
                            repeat=False)

            try:
                if self.visualize:
                    if animation_path:
                        self.export = True
                        pbar = tqdm(total=len(self), unit="frames", desc="Generating animation")
                        self.ani.save(animation_path,
                                      writer='ffmpeg',
                                      fps=self.visuals.get("fps", DEFAULT_FPS),
                                      progress_callback=lambda i, n: pbar.update(1))
                        pbar.close()
                        print(f"Animation saved to {animation_path}")
                    else:
                        self.export = False
                        plt.show()
                        if not self.done:
                            print(len(self), "frames remaining.")
                            for _ in tqdm(self, unit="frame", desc="Visual closed, running remaining simulation"):
                                pass
                            self.done = True
                else:
                    for _ in tqdm(self, unit="ms"):
                        pass
                print(f"{self.title} run complete.")

            except KeyboardInterrupt:
                print("Simulation stopped.")

    def save(self, filename: str) -> None:
        """
        Save the current state of the simulation to a new configuration file.
        This is for incremental saving of the simulation result.
        Args:
            filename (str): The path to the file where the configuration will be saved.
        """
        config_data = self.get_state()
        
        with open(filename, 'w') as f:
            json.dump(config_data, f, indent=4)
        print(f"Simulation result configuration saved to {filename}")


    # This is somewhat combersome, but it is what we have to do
    # to make sure matplotlib doesnt't do something weired when
    # we try to run multiple simulations.
    def get_state(self) -> dict:
        """
        Get the current state of the simulation. This can be used for 
        incremental saving of the simulation result without writing to a file.

        Returns:
            dict: The current state of the simulation.
        """
        config_data = self.config.copy()
        config_data["objs"] = [
            {
                "pos": self.obj_pos[i].tolist(),
                "vel": self.obj_vel[i].tolist(),
                "mass": obj["mass"],
                "name": obj.get("name", ''),
                "radius": obj.get("radius", 0)
            } for i, obj in enumerate(self.config["objs"])
        ]
        return config_data