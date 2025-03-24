import numpy as np
from scipy import fft
import matplotlib.pyplot as plt
import scipy.signal as signal


duration       = 1          # длительность сигнала в секундах 
carrier_signal = 100      # Несущая частота
mod_depth      = 0.5        # Глубина модуляции
sampling_rate  = 2**16       # Частота точек на отрезке в определение функции
modulated_freq = 2

time_signal = np.linspace(0, duration, sampling_rate * duration) 
nu = fft.fftfreq(sampling_rate * duration, d = 1 / sampling_rate)

def ModuleSignal(signal):
    return (1 + mod_depth * signal) * np.cos(2 * np.pi * carrier_signal * time_signal)

def DiodFunc(signal):
    return np.maximum(signal, 0)

def CalcCarrierFreq(signal):
    C = fft.fft(signal)
    max_A = max(C)
    freq, = [nu[i] for i in range(len(nu)) if C[i] == max_A]
    return freq

def AverageOverTime(signal, carrier_freq):
    accumulate_sum = np.cumsum(signal)
    T = int(sampling_rate / carrier_freq) # Сколько точек между пиками 
    g = [*[accumulate_sum[i] / (i+1) for i in range(T)], *[(accumulate_sum[i] - accumulate_sum[i-T]) / T * np.pi for i in range(T, duration * sampling_rate)]]
    return g

def CalcModDepth(signal, carrier_freq):
    T = int(sampling_rate / carrier_freq)
    min_A = min(signal[2*T:])
    max_A = max(signal[2*T:])
    print(f'({max_A} - {min_A})/({max_A} + {min_A})')
    return (max_A - min_A)/(max_A + min_A)

def Demodulation(signal, carrier_freq):
    mod = CalcModDepth(signal, carrier_freq)
    print(f'Calculated modulation depth = {mod}')

    return [(i -1)/mod for i in signal]


if __name__ == "__main__":
    signal_for_reciving = (np.sin(2 * np.pi * modulated_freq * time_signal))
    # signal_for_reciving = (np.sin(2 * np.pi * modulated_freq * time_signal) * time_signal * np.exp(time_signal)) ** 2
    # signal_for_reciving = signal.square(2 * np.pi * modulated_freq * time_signal, duty=0.33) + 1

    modulated_signal = ModuleSignal(signal_for_reciving)
    direct_current = DiodFunc(modulated_signal)
    carrier_freq = CalcCarrierFreq(modulated_signal)

    average_sign = AverageOverTime(direct_current, carrier_freq)
    result = Demodulation(average_sign, carrier_freq)
    
    plt.figure(figsize=(20, 10))
    plt.subplot(3, 2, 1)
    plt.plot(time_signal, signal_for_reciving)
    plt.title('Исходный сигнал')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')

    plt.subplot(3, 2, 2)
    plt.plot(time_signal, modulated_signal)
    plt.title('Модулированный сигнал')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')

    plt.subplot(3, 2, 3)
    plt.plot(time_signal, direct_current)
    plt.title('Сигнал после выпрямления диодом')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')

    plt.subplot(3, 2, 4)
    plt.plot(time_signal, average_sign, label='Усредненный сигнал')
    plt.plot(time_signal,  direct_current, '-k', label='Сигнал после выпрямления диодом', alpha=0.7)
    plt.title('Сравнение сигналов до и после усреднения')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.legend()
    


    plt.subplot(3, 2, 5)
    plt.plot(time_signal,  signal_for_reciving, '--', label='Исходный', alpha=0.7)
    plt.plot(time_signal, average_sign, label='Демодулированный')
    plt.title('Сравнение демодулированного и исходного сигналов')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.legend()

    plt.subplot(3, 2, 6)
    
    # print(freq)

    spectr = np.abs(fft.fft(modulated_signal)) / (sampling_rate * duration * carrier_signal)

    plt.plot(nu, spectr, label='Спектр')
    plt.xlim(0, 2*carrier_signal )
    plt.tight_layout()
    plt.show()

