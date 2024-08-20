import numpy as np
import matplotlib.pyplot as plt

m = 1
c = 0.8
k = 1.16
f = 0.9881
p = 0.5
x0 = 1
x1 = 0

if c * c >= 4 * k * m:
    raise ValueError("Квадрат коэффициента затухания колебаний не должен превышать или быть равным произведению массы тела и жёсткости пружины, умноженному на 4")

w = np.sqrt(4 * k * m - c * c) / (2 * m)

C1 = f * (k - m * p * p) / (c * c * p * p + (k - m * p * p) ** 2)

C2 = c * f * p / (c * c * p * p + (k - m * p * p) ** 2)

A = x0 - C1

B = (x1 + c / (2 * m) * A - C2 * p) / w

def f(t):
    return np.exp(-c / (2 * m) * t) * (A * np.cos(w * t) + B * np.sin(w * t)) + C1 * np.cos(p * t) + C2 * np.sin(p * t)

def g(t):
    return -c / (2 * m) * np.exp(-c / (2 * m) * t) * (A * np.cos(w * t) + B * np.sin(w * t)) + np.exp(-c / (2 * m) * t) * w * (B * np.cos(w * t) - A * np.sin(w * t)) + p * (C2 * np.cos(p * t) - C1 * np.sin(p * t))


plt.grid(True)

T = np.linspace(0, 30, 1024)
plt.plot(T, f(T), color="navy", label="Перемещение u(t)")
plt.plot(T, g(T), color = "red", label="Скорость u'(t)")

plt.legend(loc='best')
plt.show()