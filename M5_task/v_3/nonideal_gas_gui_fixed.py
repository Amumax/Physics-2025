
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Неидеальный газ (Леннарда–Джонса) — 5Б
"""

from ideal_gas_gui_fixed import IdealGasGUI
import argparse, math

EPSILON, SIGMA = 1.0, 12.0
RCUT2 = (2.5 * SIGMA) ** 2


class NonIdealGasGUI(IdealGasGUI):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fx = [0.0] * self.N
        self.fy = [0.0] * self.N

    def compute_forces(self):
        for i in range(self.N):
            self.fx[i] = self.fy[i] = 0.0
        for i in range(self.N):
            pi = self.particles[i]
            for j in range(i + 1, self.N):
                pj = self.particles[j]
                dx, dy = pi.x - pj.x, pi.y - pj.y
                r2 = dx * dx + dy * dy
                if 1e-4 < r2 < RCUT2:
                    inv_r2 = 1.0 / r2
                    inv_r6 = inv_r2 ** 3
                    inv_r12 = inv_r6 ** 2
                    fmag = 24 * EPSILON * inv_r2 * (
                        2 * (SIGMA**12) * inv_r12 - (SIGMA**6) * inv_r6
                    )
                    fx, fy = fmag * dx, fmag * dy
                    self.fx[i] += fx
                    self.fy[i] += fy
                    self.fx[j] -= fx
                    self.fy[j] -= fy

    def velocity_verlet(self, dt):
        for i, p in enumerate(self.particles):
            p.vx += 0.5 * dt * self.fx[i] / self.mass
            p.vy += 0.5 * dt * self.fy[i] / self.mass
            p.x += p.vx * dt
            p.y += p.vy * dt
            if p.x < self.R:
                p.x, p.vx = self.R, -p.vx
            if p.x > self.W_anim - self.R:
                p.x, p.vx = self.W_anim - self.R, -p.vx
            if p.y < self.R:
                p.y, p.vy = self.R, -p.vy
            if p.y > self.H_anim - self.R:
                p.y, p.vy = self.H_anim - self.R, -p.vy
        self.compute_forces()
        for i, p in enumerate(self.particles):
            p.vx += 0.5 * dt * self.fx[i] / self.mass
            p.vy += 0.5 * dt * self.fy[i] / self.mass

    def simulate(self, dt):
        self.compute_forces()
        self.velocity_verlet(dt)
        super().simulate(0)  # zero dt to collect impulse stats correctly


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=300)
    args = parser.parse_args()
    NonIdealGasGUI(N=args.particles).run()
