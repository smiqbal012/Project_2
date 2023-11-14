import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import odeint
import cProfile


def pdf(x, k):
    return k * np.exp(-x)


def cdf(x, k):
    return 1 - np.exp(-k * x)


def rejection_sampling(pdf, cdf, k, num_samples):
    samples = []
    while len(samples) < num_samples:
        x = -np.log(1 - np.random.rand())
        u = np.random.rand()
        if u < pdf(x, k) / (k * cdf(x, k)):
            samples.append(x)
    return samples


class Particle:
    def __init__(self):
        self.initial_mass = np.random.rand()
        self.initial_charge = np.random.uniform(-1, 1)
        self.mass = self.initial_mass
        self.charge = self.initial_charge
        self.position = -np.log(1 - np.random.rand())
        self.velocity = 0

    def acceleration(self, E):
        return self.charge * E / self.mass


class Simulation:
    def __init__(self, num_particles=1000, E=0.5, dt=0.1, T=5, k=3.0):
        self.particles = [Particle() for _ in range(num_particles)]
        self.E = E
        self.dt = dt
        self.time_steps = int(T / dt)
        self.positions_odeint = []
        self.positions_rk4 = []
        self.k = k
        self.kinetic_energy = []

    def particle_ode(self, y, t, q, m, E):
        x, v = y
        dxdt = v
        dvdt = q * E / m
        return [dxdt, dvdt]

    def rk4(self, y, t, dt, func, *args):
        k1 = dt * np.array(func(y, t, *args))
        k2 = dt * np.array(func(y + 0.5 * k1, t + 0.5 * dt, *args))
        k3 = dt * np.array(func(y + 0.5 * k2, t + 0.5 * dt, *args))
        k4 = dt * np.array(func(y + k3, t + dt, *args))
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def evolve(self):
        t = np.linspace(0, 5, self.time_steps)
        for p in self.particles:
            y0 = [p.position, p.velocity]
            sol_odeint = odeint(
                self.particle_ode, y0, t, args=(p.charge, p.mass, self.E)
            )[:, 0]
            self.positions_odeint.append(sol_odeint)

            sol_rk4 = [y0]
            for i in range(1, len(t)):
                next_step = self.rk4(
                    sol_rk4[-1],
                    t[i - 1],
                    self.dt,
                    self.particle_ode,
                    p.charge,
                    p.mass,
                    self.E,
                )
                sol_rk4.append(next_step)
            self.positions_rk4.append([pos_vel[0] for pos_vel in sol_rk4])

    def calculate_kinetic_energy(self):
        for sol in self.positions_odeint:
            velocities = np.gradient(sol, self.dt)
            kinetic_energy = 0.5 * self.particles[0].mass * velocities**2
            self.kinetic_energy.append(kinetic_energy)

    def plot_kinetic_energy(self):
        time = np.linspace(0, 5, self.time_steps)
        total_kinetic_energy = np.sum(self.kinetic_energy, axis=0)
        plt.figure(figsize=(8, 6))
        plt.plot(time, total_kinetic_energy)
        plt.title("Total Kinetic Energy vs. Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Total Kinetic Energy")
        plt.grid(True)
        plt.savefig("KineticEnergyVsTime_k3_1000.png")
        plt.show()

    def plot_positions(self):
        times_to_plot = [0, int(2.5 / self.dt), int(5 / self.dt) - 1]
        plt.figure(figsize=(15, 5))
        for i, time_idx in enumerate(times_to_plot):
            plt.subplot(1, 3, i + 1)
            positions_at_time = [sol[time_idx] for sol in self.positions_odeint]
            charges = [p.charge for p in self.particles]
            pos_positions = [
                pos for pos, charge in zip(positions_at_time, charges) if charge >= 0
            ]
            neg_positions = [
                pos for pos, charge in zip(positions_at_time, charges) if charge < 0
            ]
            plt.scatter(
                range(len(pos_positions)),
                pos_positions,
                s=1,
                color="b",
                label="Positive Charge",
            )
            plt.scatter(
                range(len(neg_positions)),
                neg_positions,
                s=1,
                color="r",
                label="Negative Charge",
            )
            plt.title(f"Particle Positions at t={time_idx*self.dt}s")
            plt.ylim(-15, 15)
            plt.xlabel("Particle Index")
            plt.ylabel("Position")
            plt.legend()
            plt.tight_layout()
        plt.tight_layout()
        plt.savefig("ParticlePositions_k3_1000.png")
        plt.show()

    def get_final_distributions(self):
        final_masses = [p.mass for p in self.particles]
        final_charges = [p.charge for p in self.particles]
        return final_masses, final_charges

    def get_initial_distributions(self):
        initial_masses = [p.initial_mass for p in self.particles]
        initial_charges = [p.initial_charge for p in self.particles]
        return initial_masses, initial_charges

    def perform_interactions(self):
        interaction_positions = rejection_sampling(
            pdf, cdf, self.k, len(self.particles)
        )
        interaction_threshold = 0.5
        for particle, position in zip(self.particles, interaction_positions):
            if position < interaction_threshold:
                particle.mass = np.random.rand()
                particle.charge = np.random.uniform(-1, 1)

    def plot_comparison(self):
        time_idx = np.abs(np.linspace(0, 5, self.time_steps) - 4.9).argmin()
        plt.figure(figsize=(12, 6))
        for i, (particle, pos_odeint, pos_rk4) in enumerate(
            zip(self.particles, self.positions_odeint, self.positions_rk4)
        ):
            if particle.charge > 0:
                plt.scatter(
                    i,
                    pos_odeint[time_idx],
                    color="b",
                    s=10,
                    label="odeint" if i == 0 else "",
                )
                plt.scatter(
                    i, pos_rk4[time_idx], color="r", s=10, label="RK4" if i == 0 else ""
                )
        plt.title("Comparison of Positively Charged Particle Positions at t=4.9s")
        plt.xlabel("Particle Index")
        plt.ylabel("Position")
        plt.legend()
        plt.ylim(0, 80)
        plt.grid(True)
        plt.savefig("PositionComparison_odeint_RK4_at_4.9s.png")
        plt.show()


if __name__ == "__main__":
    sim = Simulation(k=3.0)
    cProfile.runctx("sim.evolve()", globals(), locals(), "evolve_stats.prof")
    sim.evolve()
    sim.calculate_kinetic_energy()
    sim.plot_kinetic_energy()
    sim.plot_positions()
    sim.plot_comparison()
    sim.perform_interactions()
    final_masses, final_charges = sim.get_final_distributions()
    initial_masses, initial_charges = sim.get_initial_distributions()
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(initial_masses, bins=50, color="b", alpha=0.5, label="Initial")
    plt.hist(final_masses, bins=50, color="r", alpha=0.5, label="Final")
    plt.title("Mass Distribution")
    plt.xlabel("Mass")
    plt.ylabel("Count")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.hist(initial_charges, bins=50, color="g", alpha=0.5, label="Initial")
    plt.hist(final_charges, bins=50, color="m", alpha=0.5, label="Final")
    plt.title("Charge Distribution")
    plt.xlabel("Charge")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("MassChargeDistributions_k3_1000.png")
    plt.show()
    particle_counts = [10, 100, 500, 1000]
    runtimes = []
    for num_particles in particle_counts:
        sim = Simulation(num_particles=num_particles, k=3.0)
        start_time = time.time()
        sim.evolve()
        end_time = time.time()
        runtime = end_time - start_time
        runtimes.append(runtime)
    plt.figure(figsize=(8, 6))
    plt.plot(particle_counts, runtimes, marker="o")
    plt.title("Simulation Evolution Time vs. Number of Particles")
    plt.xlabel("Number of Particles")
    plt.ylabel("Time (seconds)")
    plt.grid(True)
    plt.savefig("SimulationTimeVsParticles.png")
    plt.show()
