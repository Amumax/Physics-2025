
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non‑ideal gas simulation (5B) using Lennard‑Jones potential (2D)
Authors: Максим Найденов, Лидия Осипчук

This demo reuses Pygame visualization from ideal_gas.py, but computes
forces between particles via Lennard‑Jones potential (12‑6).
"""

import argparse, math, random, collections
import numpy as np, pygame
from ideal_gas import kB

EPSILON = 1.0   # depth of LJ well (arbitrary units)
SIGMA   = 10.0  # finite size of molecule (in px)
CUTOFF  = 2.5 * SIGMA

class Particle:
    __slots__ = ('x','y','vx','vy','fx','fy')
    def __init__(self,x,y,vx,vy):
        self.x, self.y = x,y
        self.vx, self.vy = vx,vy
        self.fx = self.fy = 0.0

class LJSimulation:
    def __init__(self, N=500, width=800, height=600, mass=1.0, temperature=150.0, fps=60):
        self.N=N; self.W=width; self.H=height
        self.mass=mass; self.T0=temperature; self.dt=1e-2
        self.fps=fps
        self.particles=[]
        self.init_particles()
        pygame.init()
        self.screen=pygame.display.set_mode((self.W,self.H))
        self.clock=pygame.time.Clock(); self.running=True
    def init_particles(self):
        # Maxwell-Boltzmann init inside box
        sigma_v = math.sqrt(kB*self.T0/self.mass)
        for _ in range(self.N):
            x=random.uniform(0,self.W); y=random.uniform(0,self.H)
            vx=random.gauss(0,sigma_v); vy=random.gauss(0,sigma_v)
            self.particles.append(Particle(x,y,vx,vy))
    def compute_forces(self):
        # reset forces
        for p in self.particles:
            p.fx=p.fy=0.0
        for i in range(self.N):
            pi=self.particles[i]
            for j in range(i+1,self.N):
                pj=self.particles[j]
                dx=pi.x - pj.x; dy=pi.y - pj.y
                # minimum image (no wrap but reflect)
                # compute distance
                r2=dx*dx+dy*dy
                if r2 < CUTOFF*CUTOFF and r2>1e-2:
                    inv_r2 = 1.0/r2
                    inv_r6 = inv_r2**3
                    inv_r12=inv_r6*inv_r6
                    fmag = 24*EPSILON*inv_r2*(2*SIGMA**12*inv_r12 - SIGMA**6*inv_r6)
                    fx = fmag*dx; fy=fmag*dy
                    pi.fx += fx; pi.fy += fy
                    pj.fx -= fx; pj.fy -= fy
    def integrate(self):
        dt=self.dt
        for p in self.particles:
            # velocity Verlet (half step velocity update)
            p.vx += 0.5*dt*p.fx/self.mass
            p.vy += 0.5*dt*p.fy/self.mass
            p.x  += p.vx*dt
            p.y  += p.vy*dt
            # wall reflect
            if p.x<0: p.x=-p.x; p.vx=-p.vx
            if p.x>self.W: p.x=2*self.W-p.x; p.vx=-p.vx
            if p.y<0: p.y=-p.y; p.vy=-p.vy
            if p.y>self.H: p.y=2*self.H-p.y; p.vy=-p.vy
        # recompute forces with new positions
        self.compute_forces()
        for p in self.particles:
            p.vx += 0.5*dt*p.fx/self.mass
            p.vy += 0.5*dt*p.fy/self.mass
    def draw(self):
        self.screen.fill((0,0,0))
        for p in self.particles:
            pygame.draw.circle(self.screen,(200,200,50),(int(p.x),int(p.y)),3)
        pygame.display.flip()
    def run(self):
        self.compute_forces()
        while self.running:
            for event in pygame.event.get():
                if event.type==pygame.QUIT: self.running=False
            self.integrate(); self.draw(); self.clock.tick(self.fps)
        pygame.quit()
def main():
    parser=argparse.ArgumentParser(description="LJ non‑ideal gas")
    parser.add_argument("--particles",type=int,default=300)
    args=parser.parse_args()
    sim=LJSimulation(N=args.particles)
    sim.run()
if __name__=="__main__":
    main()
