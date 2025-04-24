
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ideal Gas Molecular Dynamics Simulation (2D) with Pygame visualization
---------------------------------------------------------------------
Authors: Максим Найденов, Лидия Осипчук
Date: 2025-04-24

Requirements:
    - pygame
    - numpy
    - matplotlib (optional, for live graphs)

Usage:
    $ python ideal_gas.py [--particles N] [--width W] [--height H] [--radius R]
                          [--temperature T] [--fps FPS]

Press ESC to quit simulation, SPACE to pause/resume.
After closing the Pygame window, if matplotlib is available, a plot of
pressure & temperature vs. time and the velocity distribution histogram
will be shown.

This script is intended as teaching / demonstration code and is not
highly optimized for >5000 particles, but uses a simple uniform grid
(cell-list) algorithm that allows thousands of particles in real-time
on a modern CPU.
"""

import argparse
import sys
import math
import random
import collections
import time

import numpy as np
import pygame

# matplotlib is optional – only for plots after simulation
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Physical constants (SI)
kB = 1.380649e-23  # J/K

class Particle:
    __slots__ = ('x', 'y', 'vx', 'vy')

    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy

class IdealGasSimulation:
    def __init__(self, N=1000, width=800, height=600, radius=4,
                 mass=1.0, temperature=300.0, fps=60):
        self.N = N
        self.W = width
        self.H = height
        self.R = radius
        self.mass = mass
        self.T0 = temperature
        self.fps = fps

        # Derived properties
        self.area = self.W * self.H  # treat pixel^2 as arbitrary units
        self.dt = 1.0 / self.fps  # real-time step (can scale physical units)

        # Cell list parameters for collision search
        self.cell_size = 2 * self.R * 1.25
        self.n_cells_x = int(math.ceil(self.W / self.cell_size))
        self.n_cells_y = int(math.ceil(self.H / self.cell_size))

        self.grid = [[[] for _ in range(self.n_cells_y)]
                     for _ in range(self.n_cells_x)]

        self.particles = []
        self.init_particles()

        # Statistics
        self.time = 0.0
        self.pressure_impulse = 0.0
        self.pressure_time_window = 1.0  # seconds
        self.impulse_history = collections.deque()  # (timestamp, impulse)
        self.temperature_history = []
        self.pressure_history = []
        self.time_history = []

        # Pygame init
        pygame.init()
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Ideal Gas Simulation (N={})".format(self.N))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.running = True
        self.paused = False

    # ---------------- initialization ----------------
    def init_particles(self):
        """
        Randomly place particles without overlap and give them velocities
        from Maxwell-Boltzmann distribution corresponding to temperature T0.
        """
        # Standard deviation of velocity components (2D)
        sigma = math.sqrt(kB * self.T0 / self.mass)
        attempts_limit = 10000
        for i in range(self.N):
            placed = False
            attempts = 0
            while not placed and attempts < attempts_limit:
                x = random.uniform(self.R, self.W - self.R)
                y = random.uniform(self.R, self.H - self.R)

                # Check overlap with existing
                overlap = False
                for p in self.particles:
                    if (p.x - x)**2 + (p.y - y)**2 < (2*self.R)**2:
                        overlap = True
                        break
                if not overlap:
                    # Maxwell-Boltzmann velocities
                    vx = random.gauss(0, sigma)
                    vy = random.gauss(0, sigma)
                    self.particles.append(Particle(x, y, vx, vy))
                    placed = True
                attempts += 1
            if attempts == attempts_limit:
                raise RuntimeError("Could not place all particles without overlap. "
                                   "Try smaller N or radius.")
        # Remove net momentum (center-of-mass rest)
        vx_cm = sum(p.vx for p in self.particles)/self.N
        vy_cm = sum(p.vy for p in self.particles)/self.N
        for p in self.particles:
            p.vx -= vx_cm
            p.vy -= vy_cm

    # ---------------- simulation loop ----------------
    def run(self):
        while self.running:
            self.handle_events()
            if not self.paused:
                self.simulate_step()
            self.draw()
            self.clock.tick(self.fps)
        pygame.quit()
        self.post_process_plots()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused

    # ---------------- physics ----------------
    def simulate_step(self):
        dt = self.dt
        self.time += dt

        # Clear spatial grid
        for cx in range(self.n_cells_x):
            for cy in range(self.n_cells_y):
                self.grid[cx][cy].clear()

        # Move particles & insert into grid
        for idx, p in enumerate(self.particles):
            p.x += p.vx * dt
            p.y += p.vy * dt

            # Wall collisions (left/right)
            if p.x < self.R:
                self.add_impulse(2*self.mass*abs(p.vx))
                p.x = self.R
                p.vx = -p.vx
            elif p.x > self.W - self.R:
                self.add_impulse(2*self.mass*abs(p.vx))
                p.x = self.W - self.R
                p.vx = -p.vx
            # Wall collisions (top/bottom)
            if p.y < self.R:
                self.add_impulse(2*self.mass*abs(p.vy))
                p.y = self.R
                p.vy = -p.vy
            elif p.y > self.H - self.R:
                self.add_impulse(2*self.mass*abs(p.vy))
                p.y = self.H - self.R
                p.vy = -p.vy

            # Insert into cell
            cx = int(p.x / self.cell_size)
            cy = int(p.y / self.cell_size)
            self.grid[cx][cy].append(idx)

        # Handle particle collisions (cell-list algorithm)
        for cx in range(self.n_cells_x):
            for cy in range(self.n_cells_y):
                # Check collisions within this cell and neighboring cells
                neighbor_cells = [(cx+dx, cy+dy) for dx in (-1,0,1) for dy in (-1,0,1)
                                  if 0 <= cx+dx < self.n_cells_x and 0 <= cy+dy < self.n_cells_y]
                for (nx, ny) in neighbor_cells:
                    for i in self.grid[cx][cy]:
                        for j in self.grid[nx][ny]:
                            if i >= j:
                                continue
                            self.resolve_collision(i, j)

        # Record stats periodically
        if int(self.time / self.pressure_time_window) > len(self.pressure_history):
            self.record_stats()

        # Expire old impulse records
        while self.impulse_history and self.time - self.impulse_history[0][0] > self.pressure_time_window:
            self.pressure_impulse -= self.impulse_history[0][1]
            self.impulse_history.popleft()

    def resolve_collision(self, i, j):
        p1 = self.particles[i]
        p2 = self.particles[j]
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        dist2 = dx*dx + dy*dy
        min_dist = 2 * self.R
        if dist2 < min_dist * min_dist and dist2 > 0:
            dist = math.sqrt(dist2)
            nx = dx / dist
            ny = dy / dist

            # Relative velocity
            dvx = p1.vx - p2.vx
            dvy = p1.vy - p2.vy
            v_rel = dvx * nx + dvy * ny
            if v_rel > 0:
                return  # moving apart

            # Impulse magnitude
            J = -(1 + 1) * v_rel / 2  # equal masses

            # Apply impulse
            p1.vx += -J * nx
            p1.vy += -J * ny
            p2.vx += J * nx
            p2.vy += J * ny

            # Separate overlap
            overlap = min_dist - dist
            p1.x -= nx * overlap / 2
            p1.y -= ny * overlap / 2
            p2.x += nx * overlap / 2
            p2.y += ny * overlap / 2

    def add_impulse(self, delta_p):
        self.pressure_impulse += delta_p
        self.impulse_history.append((self.time, delta_p))

    def record_stats(self):
        # temperature
        v2 = sum(p.vx*p.vx + p.vy*p.vy for p in self.particles) / self.N
        T_inst = self.mass * v2 / (2 * kB)
        # pressure (force/length here) convert to effective pressure using area
        pressure_inst = self.pressure_impulse / (self.pressure_time_window * (2*(self.W+self.H)))
        self.temperature_history.append(T_inst)
        self.pressure_history.append(pressure_inst)
        self.time_history.append(self.time)

    # ---------------- drawing ----------------
    def draw(self):
        self.screen.fill((0, 0, 0))
        # Draw particles
        for p in self.particles:
            speed = math.hypot(p.vx, p.vy)
            # Map speed to color (blue->green->red)
            c = min(255, int(speed * 30))
            color = (c, 255-c, 128)
            pygame.draw.circle(self.screen, color, (int(p.x), int(p.y)), self.R)

        # Render text
        info_text = "t = {:6.2f} s   Particles: {}   Press SPACE to pause".format(self.time, self.N)
        text_surface = self.font.render(info_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))

        pygame.display.flip()

    # -------------- post-process plots --------------
    def post_process_plots(self):
        if not HAS_MPL or len(self.time_history) < 2:
            return
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(8, 6))
        ax1.plot(self.time_history, self.temperature_history, label="T (K)")
        ax1.set_ylabel("Temperature (K)")
        ax1.grid(True)
        ax1.legend()

        ax2.plot(self.time_history, self.pressure_history, label="P (model units)")
        ax2.set_ylabel("Pressure (impulse/(length*time))")
        ax2.set_xlabel("Time (s)")
        ax2.grid(True)
        ax2.legend()

        # Velocity distribution histogram at final time
        speeds = [math.hypot(p.vx, p.vy) for p in self.particles]
        fig2, ax3 = plt.subplots(figsize=(6,4))
        counts, bins, _ = ax3.hist(speeds, bins=30, density=True, alpha=0.6, label="Simulated")
        # Theoretical Maxwell-Boltzmann (2D)
        v = np.linspace(0, max(speeds), 200)
        T_final = self.temperature_history[-1]
        f_mb = (self.mass * v / (kB*T_final)) * np.exp(- self.mass * v**2 / (2*kB*T_final))
        ax3.plot(v, f_mb, 'r--', label="Maxwell–Boltzmann")
        ax3.set_xlabel("Speed")
        ax3.set_ylabel("Probability density")
        ax3.legend()
        ax3.set_title("Velocity distribution")

        plt.tight_layout()
        plt.show()

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser(description="Ideal Gas MD simulation (2D, Pygame)")
    parser.add_argument("--particles", type=int, default=1000, help="Number of molecules (default 1000)")
    parser.add_argument("--width", type=int, default=800, help="Container width in px")
    parser.add_argument("--height", type=int, default=600, help="Container height in px")
    parser.add_argument("--radius", type=int, default=4, help="Particle radius in px")
    parser.add_argument("--temperature", type=float, default=300.0, help="Initial temperature (K)")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second")
    args = parser.parse_args()

    sim = IdealGasSimulation(N=args.particles,
                             width=args.width,
                             height=args.height,
                             radius=args.radius,
                             temperature=args.temperature,
                             fps=args.fps)
    sim.run()

if __name__ == "__main__":
    main()
