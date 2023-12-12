import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
import matplotlib.ticker

a = 0
b = 1.5
epsilon = np.logspace(-10, -1, 10)

# Задаем функцию
def f(x):
    return np.sin(x) / np.sqrt(1 + 2 * np.sin(x)**2)

# Вычисление первообразной
F = lambda x: -np.arcsin(min(1, np.sqrt(2/3))) / (np.sqrt(2) * np.cos(x))

# Коэффициенты метода Гаусса
A = np.array([0.5, 0.5])
X = np.array([0.2113249, 0.7886751])

def gauss(eps):
    ret = np.zeros(3)
    s, _ = spi.quad(f, a, b)
    h = b - a
    ys = s
    t = 0
    while t < 9999:
        y = 0
        t = 0
        x_val = a
        while x_val < (b - (h / 2)):
            for i in range(0, len(A)):
                y = y + A[i] * f(x_val + h * X[i])
                t = t + 1
            x_val = x_val + h
        y = y * h
        if (np.abs((y - ys) / ys)) < eps:
            break
        ys = y
        h = h / 2
    ret[0] = abs(s - y)
    ret[1] = t
    ret[2] = y
    return ret

def simpson(eps):
    ret = np.zeros(3)
    s, _ = spi.quad(f, a, b)
    h = (b - a) / (2 * len(A))
    ys = s
    t = 0
    while t < 9999:
        y = 0
        t = 0
        x_val = a
        while x_val < (b - h):
            y = y + f(x_val) + 4 * f(x_val + h) + f(x_val + 2 * h)
            t = t + 1
            x_val = x_val + 2 * h
        y = y * (h / 3)
        if (np.abs((y - ys) / ys)) < eps:
            break
        ys = y
        h = h / 2
    ret[0] = abs(y - s)
    ret[1] = t
    ret[2] = y
    return ret

# Вычисление значений для графиков
Z = [gauss(eps) for eps in epsilon]
C = [simpson(eps) for eps in epsilon]
F_vals = [F(i) for i in np.arange(a, b, 0.1)]

# Построение графиков
plt.figure(figsize=(10, 6))

# Первый график - функция и первообразная
plt.subplot(2, 1, 1)
plt.title("Функция и первообразная")
plt.plot(np.arange(a, b, 0.1), [F(i) for i in np.arange(a, b, 0.1)], label='F(x)')
plt.plot(np.arange(a, b, 0.1), [f(i) for i in np.arange(a, b, 0.1)], label='f(x)')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()

plt.figure(figsize=(10, 6))
plt.title("График приближения значения от точности")
print("Gauss values:", [Z[i][2] for i in range(10)])
print("Simpson values:", [C[i][2] for i in range(10)])
plt.scatter(epsilon, [Z[i][2] for i in range(10)], label='Метод Гаусса')
plt.scatter(epsilon, [C[i][2] for i in range(10)], marker='x', label='Метод Симпсона')
plt.plot(epsilon, [F(b) - F(a) for _ in epsilon], linestyle='dotted', label='Значение по Ньютону-Лейбницу')
plt.xscale('log')
plt.xlabel("Точность")
plt.ylabel("Приближенное значение")
plt.legend()
plt.grid()

# Добавляем метки к точкам для метода Гаусса и метода Симпсона
for i, txt in enumerate(["{:.10f}".format(Z[i][2]) for i in range(10)]):
    plt.annotate(txt, (epsilon[i], Z[i][2]), textcoords="offset points", xytext=(-10,0), ha='center', fontsize=8)

for i, txt in enumerate(["{:.10f}".format(C[i][2]) for i in range(10)]):
    plt.annotate(txt, (epsilon[i], C[i][2]), textcoords="offset points", xytext=(-10,0), ha='center', fontsize=8)

plt.show()

# Третий график - количество итераций от точности
plt.figure(figsize=(10, 6))
plt.title("График количества итераций от точности")
plt.semilogx(epsilon, [Z[i][1] for i in range(10)], label='Метод Гаусса')
plt.semilogx(epsilon, [C[i][1] for i in range(10)], label='Метод Симпсона')
plt.xlabel("Точность")
plt.ylabel("Количество итераций")
plt.legend()
plt.grid()
plt.show()

# Четвертый график - фактическая ошибка интегрирования от точности
plt.figure(figsize=(10, 6))
plt.title("График фактической ошибки интегрирования от точности")
plt.loglog(epsilon, [Z[i][0] for i in range(10)], label='Метод Гаусса')
plt.loglog(epsilon, [C[i][0] for i in range(10)], linestyle='--', label='Метод Симпсона')
plt.xlabel("Точность")
plt.ylabel("Фактическая ошибка интегрирования")
plt.legend()
plt.grid()
plt.show()
