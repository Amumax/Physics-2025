
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adiabatic Gas Process (5A) Simulation
-------------------------------------
Extends ideal_gas.py to include a moving right wall (piston).
The piston moves slowly inward/outward to realise quasi-static
adiabatic compression/expansion. Wall is adiabatic: collisions are
still perfectly elastic, so no heat exchange; volume changes cause
work on/of the gas, changing its temperature.

Run:
    $ python adiabatic_gas.py --compress    (slow compression)
    $ python adiabatic_gas.py --expand      (slow expansion)
    $ python adiabatic_gas.py --free_expand (instant expansion)
"""

import argparse, math, time, pygame, numpy as np
from ideal_gas import IdealGasSimulation, kB

class AdiabaticSimulation(IdealGasSimulation):
    def __init__(self, mode='compress', duration=10.0, target_scale=0.5, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.duration = duration
        self.target_scale = target_scale
        self.initial_width = self.W
        self.initial_height = self.H
        self.elapsed = 0.0

    def simulate_step(self):
        # Update piston position before physics
        if self.mode in ('compress','expand') and self.elapsed < self.duration:
            frac = self.elapsed / self.duration
            if self.mode == 'compress':
                scale = 1.0 - frac*(1-self.target_scale)
            else: # expand
                scale = 1.0 + frac*(self.target_scale-1)
            new_W = int(self.initial_width * scale)
            # Move right wall
            self.W = new_W
            # Make sure particles inside
            for p in self.particles:
                if p.x > self.W - self.R:
                    p.x = self.W - self.R
                    p.vx = -abs(p.vx)
        elif self.mode == 'free_expand' and self.elapsed == 0.0:
            # instant doubling of volume to the right
            self.W = int(self.initial_width * (1+self.target_scale))
        self.elapsed += self.dt
        super().simulate_step()

def main():
    parser = argparse.ArgumentParser(description="Adiabatic process simulation (2D)")
    parser.add_argument("--mode", choices=["compress","expand","free_expand"], default="compress")
    parser.add_argument("--duration", type=float, default=10.0, help="Time for compression/expansion")
    parser.add_argument("--target_scale", type=float, default=0.5, help="Final width factor (compress <1, expand >1)")
    parser.add_argument("--particles", type=int, default=1000)
    args = parser.parse_args()

    sim = AdiabaticSimulation(mode=args.mode,
                              duration=args.duration,
                              target_scale=args.target_scale,
                              N=args.particles)
    sim.run()

if __name__ == "__main__":
    main()
