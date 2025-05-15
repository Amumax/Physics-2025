import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import matplotlib.animation as animation


class Params:
    nx, ny    = 77, 77     # размер сетки
    dt        = 0.02         # шаг по времени
    kappa     = 0.1          # теплопроводность
    C         = 1.0          # теплоёмкость
    T_pl      = 0.0          # температура плавления
    gamma     = 0.5          # чувствительность переохлаждения
    L         = 1.0          # теплота плавления
    steps     = 5000         # число шагов
    interval  = 50           # ms между кадрами

p = Params()

# Начальные условия
T = np.random.uniform(low=p.T_pl - 10.0, high=p.T_pl + 5.0, size=(p.nx, p.ny))
ice = np.zeros((p.nx, p.ny), bool)

# вспомогалки
def neighbors(i,j):
    for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
        ii, jj = i+di, j+dj
        if 0 <= ii < p.nx and 0 <= jj < p.ny:
            yield ii, jj

rng = np.random.default_rng(42)

# Обновляем шаг
def step():
    global T, ice
    # обмен теплотой
    dT = np.zeros_like(T)
    for i in range(p.nx):
        for j in range(p.ny):
            for ii,jj in neighbors(i,j):
                dT[i,j] += T[ii,jj] - T[i,j]
    T += p.kappa * p.dt / p.C * dT

    # замерзание
    for i in range(p.nx):
        for j in range(p.ny):
            if not ice[i,j] and T[i,j] < p.T_pl:
                nIce = sum(ice[ii,jj] for ii,jj in neighbors(i,j))
                P = (1 - np.exp(-p.gamma*(p.T_pl - T[i,j]))) * (nIce/4) # вероятность что замерзнет
                if rng.random() < P:
                    ice[i,j] = True
                    # забираем у соседей теплоту плавления
                    for ii,jj in neighbors(i,j):
                        T[ii,jj] -= p.L/(p.C*4)

fig, ax = plt.subplots()
im = ax.imshow(T, cmap='coolwarm', vmin=p.T_pl-2, vmax=T.max())
cbar = fig.colorbar(im, ax=ax)


def update(frame):
    for _ in range(5):
        step()
    display = np.where(ice, p.T_pl, T)
    im.set_data(display)
    return im,

ani = animation.FuncAnimation(
    fig, update, frames=int(p.steps/5), 
    interval=p.interval, blit=True
)

plt.show()
