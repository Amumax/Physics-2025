
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Газ в однородном поле тяжести (5В)
"""
from ideal_gas_gui_fixed import IdealGasGUI
import argparse


class GravityGasGUI(IdealGasGUI):
    def __init__(self, g=0.25, **kwargs):
        super().__init__(**kwargs)
        self.g = g  # px/кадр²

    def simulate(self, dt):
        for p in self.particles:
            p.vy += self.g * dt
        super().simulate(dt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=400)
    parser.add_argument("--g", type=float, default=0.25)
    args = parser.parse_args()
    GravityGasGUI(N=args.particles, g=args.g).run()
