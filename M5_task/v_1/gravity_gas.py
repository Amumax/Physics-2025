
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ideal gas in gravity field (5В) Simulation
------------------------------------------
Extends ideal_gas.py by adding constant downward acceleration g.
"""

import argparse, math, random, pygame, numpy as np
from ideal_gas import IdealGasSimulation, kB

G_ACCEL = 9.81e-6  # acceleration per frame unit (arbitrary scaling)

class GravitySimulation(IdealGasSimulation):
    def __init__(self, g=G_ACCEL, **kwargs):
        super().__init__(**kwargs)
        self.g = g
    def simulate_step(self):
        # add gravity to velocities
        for p in self.particles:
            p.vy += self.g * self.dt
        super().simulate_step()
    def draw(self):
        super().draw()
        # maybe draw gradient background or scale colors

def main():
    parser = argparse.ArgumentParser(description="Gas column in gravity field")
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--g", type=float, default=G_ACCEL)
    args = parser.parse_args()
    sim = GravitySimulation(N=args.particles, g=args.g)
    sim.run()

if __name__ == "__main__":
    main()
