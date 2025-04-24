
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Адиабатические процессы (5A) — сжатие / расширение
"""

from ideal_gas_gui_fixed import IdealGasGUI
import argparse, pygame


class AdiabaticGasGUI(IdealGasGUI):
    def __init__(self, mode="compress", duration=10.0, target_scale=0.5, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.duration = duration
        self.target_scale = target_scale
        self.elapsed = 0.0
        self.W_start = self.W_anim

    def adjust_volume(self, dt):
        self.elapsed += dt
        if self.mode == "free_expand" and self.elapsed < dt * 1.5:
            self.W_anim = int(self.W_start * (1 + self.target_scale))
        elif self.mode in ("compress", "expand") and self.elapsed < self.duration:
            frac = self.elapsed / self.duration
            if self.mode == "compress":
                scale = 1.0 - frac * (1 - self.target_scale)
            else:
                scale = 1.0 + frac * (self.target_scale - 1)
            self.W_anim = int(self.W_start * scale)
        # частицам не позволять выйти
        for p in self.particles:
            if p.x > self.W_anim - self.R:
                p.x = self.W_anim - self.R
                p.vx = -abs(p.vx)

    def simulate(self, dt):
        self.adjust_volume(dt)
        super().simulate(dt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["compress", "expand", "free_expand"], default="compress")
    parser.add_argument("--particles", type=int, default=400)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--scale", type=float, default=0.5)
    args = parser.parse_args()
    AdiabaticGasGUI(
        mode=args.mode, duration=args.duration, target_scale=args.scale, N=args.particles
    ).run()
