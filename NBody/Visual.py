import numpy as np

COLORS = np.array(['r', 'g', 'b', 'c', 'm', 'y', 'k'])
TRACE_LENGTH = 50

class Visualizer(object):
    def __init__(self, solver) -> None:
        self.solver = solver

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.animation import FuncAnimation


        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.set_scope(solver.scope)
        self.history = []
        self.colors = np.random.choice(COLORS, size=(solver.obj_pos.shape[0]))
        self.lines = [self.ax.plot([], [], [], color = self.colors[i])[0] for i in range(solver.obj_pos.shape[0])]
        self.scat = self.ax.scatter(solver.obj_pos[:, 0], solver.obj_pos[:, 1], solver.obj_pos[:, 2], c=self.colors, marker='o')
        self.text_box = self.ax.text2D(-0.1, 0.9, '', transform=self.ax.transAxes)
        self.ani = FuncAnimation(self.fig, self.update, frames=self.solver, interval=self.solver.ms_per_frame, blit=False)
        plt.show()
        
    def set_scope(self, scope) -> None:
        self.ax.set_xlim3d(scope[0])
        self.ax.set_ylim3d(scope[1])
        self.ax.set_zlim3d(scope[2])
        self.ax.set_title("3D Animated Scatter Plot")
    
    def update(self, solver) -> None:
        if self.solver.tsim_current_ms % self.solver.ms_per_frame == 0:
            self.text_box.set_text(fr"simulation time: {solver.tsim_current:.2f} s" + \
                                   '\n' + fr"$\Delta E = {solver.delta_E:.2f} J$" + \
                                    f"({solver.delta_E / solver.E:.1f}%)")

            self.scat._offsets3d = (solver.obj_pos[:, 0], solver.obj_pos[:, 1], solver.obj_pos[:, 2])
                # Get the updated coordinates
            self.history.append(solver.obj_pos.copy())
            
            # Limit history length for trace
            if len(self.history) > TRACE_LENGTH:
                self.history.pop(0)
                
            # Update trace lines
            for i in range(solver.obj_pos.shape[0]):
                trace_data = np.array(self.history)[:, i, :]
                self.lines[i].set_data(trace_data[:, 0], trace_data[:, 1])
                self.lines[i].set_3d_properties(trace_data[:, 2])
    
        return self.scat, *self.lines, self.text_box