import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Particle Class
class Particle:
    def __init__(self):
        self.mass = np.random.rand()          # Randomly choose mass between 0 and 1.
        self.charge = np.random.uniform(-1, 1) # Randomly choose charge between -1 and 1.
        u = np.random.rand()
        self.position = -np.log(1 - u)        # Using the inverse transform sampling method.
        self.velocity = 0                     # Initial velocity is set to 0.

    def acceleration(self, E):
        return self.charge * E / self.mass    # Acceleration due to electric field.

# Simulation Class
class Simulation:
    def __init__(self, num_particles=1000, E=0.5, dt=0.1, T=5):  # E and T have been modified here.
        self.particles = [Particle() for _ in range(num_particles)]
        self.E = E
        self.dt = dt
        self.time_steps = int(T/dt)
        self.positions_odeint = []

    def particle_ode(self, y, t, q, m, E):
        x, v = y
        dxdt = v
        dvdt = q * E / m
        return [dxdt, dvdt]

    def evolve(self):
        t = np.linspace(0, 5, self.time_steps)  # Updated the end time for linspace
        
        self.positions_odeint = [odeint(self.particle_ode, [p.position, p.velocity], t, args=(p.charge, p.mass, self.E))[:,0] for p in self.particles]

    def plot_positions(self):
        times_to_plot = [0, int(2.5/self.dt), int(5/self.dt) - 1]  # Changed times to match new total time
        
        plt.figure(figsize=(15, 5))
        for i, time_idx in enumerate(times_to_plot):
            plt.subplot(1, 3, i+1)
            positions_at_time = [sol[time_idx] for sol in self.positions_odeint]
            plt.scatter(range(len(positions_at_time)), positions_at_time, s=1)  # Scatter plot to show individual particle positions.
            plt.title(f"Particle Positions at t={time_idx*self.dt}s")
            plt.xlabel("Particle Index")
            plt.ylabel("Position")
            plt.tight_layout()

        plt.tight_layout()
        plt.show()

# Main Execution
if __name__ == "__main__":
    sim = Simulation()
    sim.evolve()
    sim.plot_positions()

