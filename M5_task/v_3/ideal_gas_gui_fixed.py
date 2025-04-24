
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ideal Gas Simulation with GUI, polzunok speed & live graphs
"""

import math, random, collections
import pygame, pygame.freetype
pygame.init()
pygame.freetype.init()

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

kB_sim = 1.0  # "постоянная Больцмана" в px^2/кадр^2


class Particle:
    __slots__ = ("x", "y", "vx", "vy")

    def __init__(self, x, y, vx, vy):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy


class IdealGasGUI:
    def __init__(self, N=400, width=600, height=600, radius=4, T=300, fps=60):
        self.N = N
        self.W_anim, self.H_anim = width, height
        self.R = radius
        self.PANEL_W = 450
        self.screen = pygame.display.set_mode(
            (self.W_anim + self.PANEL_W, self.H_anim + 140)
        )
        pygame.display.set_caption("Идеальный газ")
        self.font = pygame.freetype.SysFont(None, 20)
        self.clock = pygame.time.Clock()

        # time
        self.dt_base = 1.0 / fps
        self.speed_scale = 1.0
        self.time = 0.0
        self.mass = 1.0

        # particles
        self.particles = []
        self.init_particles(T)

        # stats
        self.temp_hist, self.press_hist, self.t_hist = [], [], []
        self.impulse_window = collections.deque()

        # slider
        self.slider_rect = pygame.Rect(60, self.H_anim + 90, 300, 8)
        self.knob_x = self.slider_rect.centerx
        self.drag = False

        # spatial grid
        self.cell_size = 2 * self.R * 1.3
        self.grid_w = math.ceil(self.W_anim / self.cell_size)
        self.grid_h = math.ceil(self.H_anim / self.cell_size)
        self.grid = [[] for _ in range(self.grid_w * self.grid_h)]

    # -----------------------------------------------------------------
    def init_particles(self, T):
        sigma = math.sqrt(kB_sim * T / self.mass)
        while len(self.particles) < self.N:
            x = random.uniform(self.R, self.W_anim - self.R)
            y = random.uniform(self.R, self.H_anim - self.R)
            if all((p.x - x) ** 2 + (p.y - y) ** 2 > (2 * self.R) ** 2 for p in self.particles):
                vx, vy = random.gauss(0, sigma), random.gauss(0, sigma)
                self.particles.append(Particle(x, y, vx, vy))
        # убрать общий импульс
        vx_cm = sum(p.vx for p in self.particles) / self.N
        vy_cm = sum(p.vy for p in self.particles) / self.N
        for p in self.particles:
            p.vx -= vx_cm
            p.vy -= vy_cm

    # -----------------------------------------------------------------
    def grid_idx(self, x, y):
        return int(x / self.cell_size) + int(y / self.cell_size) * self.grid_w

    # -----------------------------------------------------------------
    def simulate(self, dt):
        self.time += dt
        for cell in self.grid:
            cell.clear()

        impulse = 0.0

        # move + wall collisions
        for idx, p in enumerate(self.particles):
            p.x += p.vx * dt
            p.y += p.vy * dt
            if p.x < self.R:
                p.x = self.R
                p.vx = -p.vx
                impulse += 2 * self.mass * abs(p.vx)
            elif p.x > self.W_anim - self.R:
                p.x = self.W_anim - self.R
                p.vx = -p.vx
                impulse += 2 * self.mass * abs(p.vx)
            if p.y < self.R:
                p.y = self.R
                p.vy = -p.vy
                impulse += 2 * self.mass * abs(p.vy)
            elif p.y > self.H_anim - self.R:
                p.y = self.H_anim - self.R
                p.vy = -p.vy
                impulse += 2 * self.mass * abs(p.vy)
            self.grid[self.grid_idx(p.x, p.y)].append(idx)

        # pressure impulse
        self.impulse_window.append((self.time, impulse))
        while self.impulse_window and self.time - self.impulse_window[0][0] > 1.0:
            self.impulse_window.popleft()

        # particle collisions
        for cell_idx, cell in enumerate(self.grid):
            cx, cy = cell_idx % self.grid_w, cell_idx // self.grid_w
            neigh = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                        neigh.extend(self.grid[nx + ny * self.grid_w])
            for i in cell:
                for j in neigh:
                    if j <= i:
                        continue
                    self.handle_col(i, j)

    def handle_col(self, i, j):
        p1, p2 = self.particles[i], self.particles[j]
        dx, dy = p2.x - p1.x, p2.y - p1.y
        r2 = dx * dx + dy * dy
        min_d = 2 * self.R
        if r2 < min_d * min_d and r2 > 0:
            dist = math.sqrt(r2)
            nx, ny = dx / dist, dy / dist
            dvx, dvy = p1.vx - p2.vx, p1.vy - p2.vy
            vrel = dvx * nx + dvy * ny
            if vrel > 0:
                return
            J = -2 * vrel / 2  # равные массы
            p1.vx += -J * nx
            p1.vy += -J * ny
            p2.vx += J * nx
            p2.vy += J * ny
            overlap = min_d - dist
            p1.x -= nx * overlap / 2
            p1.y -= ny * overlap / 2
            p2.x += nx * overlap / 2
            p2.y += ny * overlap / 2

    # -----------------------------------------------------------------
    def collect_stats(self):
        v2 = sum(p.vx * p.vx + p.vy * p.vy for p in self.particles) / self.N
        T = v2 / 2  # kB_sim=1
        impulse = sum(i for _, i in self.impulse_window)
        P = impulse / (2 * (self.W_anim + self.H_anim))
        self.t_hist.append(self.time)
        self.temp_hist.append(T)
        self.press_hist.append(P)

    # -----------------------------------------------------------------
    def draw(self, paused=False):
        self.screen.fill((25, 25, 35))
        # animation area
        pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, self.W_anim, self.H_anim))
        for p in self.particles:
            pygame.draw.circle(self.screen, (0, 180, 255), (int(p.x), int(p.y)), self.R)
        pygame.draw.rect(self.screen, (180, 180, 180), (0, 0, self.W_anim, self.H_anim), 1)

        # slider
        pygame.draw.rect(self.screen, (120, 120, 120), self.slider_rect)
        pygame.draw.circle(self.screen, (250, 60, 60), (self.knob_x, self.slider_rect.centery), 9)
        self.font.render_to(
            self.screen,
            (self.slider_rect.x, self.slider_rect.y - 24),
            f"Скорость: {self.speed_scale:.2f}×",
            (255, 255, 255),
        )

        # caption
        self.font.render_to(
            self.screen,
            (10, self.H_anim + 10),
            "Идеальный газ: упругие столкновения, P·V = N k_B T",
            (255, 255, 255),
        )

        # graphs
        if HAS_MPL and self.t_hist:
            fig = plt.figure(figsize=(4, 6))
            ax1 = fig.add_subplot(311)
            ax1.plot(self.t_hist, self.temp_hist, color="orange")
            ax1.set_title("Температура")
            ax2 = fig.add_subplot(312)
            ax2.plot(self.t_hist, self.press_hist, color="cyan")
            ax2.set_title("Давление")
            ax3 = fig.add_subplot(313)
            ax3.hist([math.hypot(p.vx, p.vy) for p in self.particles], bins=15, color="lime")
            ax3.set_title("Скорости")
            for ax in (ax1, ax2, ax3):
                ax.tick_params(labelsize=6)
            fig.tight_layout()
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            raw = canvas.buffer_rgba()
            w, h = canvas.get_width_height()
            surf = pygame.image.frombuffer(raw, (w, h), "RGBA")
            self.screen.blit(surf, (self.W_anim + 20, 10))
            plt.close(fig)

        if paused:
            self.font.render_to(
                self.screen, (self.W_anim // 2 - 30, self.H_anim // 2), "Пауза", (255, 0, 0)
            )
        pygame.display.flip()

    # -----------------------------------------------------------------
    def run(self):
        running, paused = True, False
        stat_every = 0.2
        next_stat = stat_every
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.type == pygame.MOUSEBUTTONDOWN:
                    if e.button == 1 and self.slider_rect.collidepoint(e.pos):
                        self.drag = True
                elif e.type == pygame.MOUSEBUTTONUP:
                    self.drag = False
                elif e.type == pygame.MOUSEMOTION and self.drag:
                    self.knob_x = max(
                        self.slider_rect.x,
                        min(e.pos[0], self.slider_rect.x + self.slider_rect.width),
                    )
                    rel = (self.knob_x - self.slider_rect.x) / self.slider_rect.width
                    self.speed_scale = 0.1 + rel * (5.0 - 0.1)

            if not paused:
                dt = self.dt_base * self.speed_scale
                self.simulate(dt)
                if self.time >= next_stat:
                    next_stat += stat_every
                    self.collect_stats()

            self.draw(paused)
            self.clock.tick(60)
        pygame.quit()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=400)
    parser.add_argument("--temp", type=float, default=300.0)
    args = parser.parse_args()
    IdealGasGUI(N=args.particles, T=args.temp).run()
