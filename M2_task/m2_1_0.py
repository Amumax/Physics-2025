import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 200                     # Количество узлов
L = 1.0                     # Индуктивность
C = 0.01                    # Ёмкость
Z = np.sqrt(L / C)          # Волновое сопротивление
dt = 0.9 * np.sqrt(L * C)   # Шаг времени
steps = 500                 # Число шагов
w = 0.2                     # Частота источника
speed = 2                   # Для красивой анимации
R = 5                       # Сопротивление контура

case = 2

if case == 1: # Случай 1: Замкнутый накоротко конец
    R = 0
elif case == 2: # Случай 2: Согласованная нагрузка
    R = Z
elif case == 3: # Случай 3: Большая dt
    dt = 0.1
    R = 0

V = np.zeros(N + 1)
I = np.zeros(N)

fig, ax = plt.subplots()
line, = ax.plot(V)
ax.set_ylim(-1, 1)
ax.set_xlabel('Узел')
ax.set_ylabel('Напряжение')


def animate(frame):
    global V, I
    for i in range(speed):
        V[0] = 0.5 * np.sin(w * (frame * speed + i))

        V_new = V.copy()
        V_new[1:-1] += (I[:-1] - I[1:]) / C * dt
        V_new[0] += (-I[0]) / C * dt  # Левый конец
        V_new[-1] = I[-1] * R  # Нагрука

        dI = (V_new[:-1] - V_new[1:]) / L * dt
        I += dI

        V = V_new
    line.set_ydata(V)
    return line,


ani = FuncAnimation(fig, animate, frames=10000, interval=50, blit=True)
plt.show()