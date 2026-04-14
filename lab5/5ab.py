import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# ФУНКЦІЯ
# ----------------------------
def f(x):
    return 50 + 20*np.sin(np.pi * x / 12) + 5*np.exp(-0.2 * (x - 12)**2)

a, b = 0, 24

# ----------------------------
# СІМПСОН
# ----------------------------
def simpson(a, b, n):
    if n % 2 != 0:
        n += 1

    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)

    S = y[0] + y[-1]

    for i in range(1, n):
        if i % 2 == 0:
            S += 2*y[i]
        else:
            S += 4*y[i]

    return S * h / 3

# ----------------------------
# "ТОЧНЕ" ЗНАЧЕННЯ
# ----------------------------
I_exact = simpson(a, b, 2000)

# ----------------------------
# ТАБЛИЦЯ + ПОХИБКА
# ----------------------------
n_values = [10, 20, 40, 80, 160, 320]
errors = []
table_text = "n      I              error\n"

for n in n_values:
    I = simpson(a, b, n)
    error = abs(I_exact - I)
    errors.append(error)
    table_text += f"{n:<6}{I:.6f}     {error:.6f}\n"

# ----------------------------
# РУНГЕ-РОМБЕРГ
# ----------------------------
I1 = simpson(a, b, 40)
I2 = simpson(a, b, 80)
I_rr = I2 + (I2 - I1)/15
error_rr = abs(I_exact - I_rr)

runge_text = f"""
Runge-Romberg:
I1 = {I1:.6f}
I2 = {I2:.6f}
I_rr = {I_rr:.6f}
error = {error_rr:.6f}
"""

# ----------------------------
# ЕЙТКЕН
# ----------------------------
I1 = simpson(a, b, 20)
I2 = simpson(a, b, 40)
I3 = simpson(a, b, 80)

p = np.log(abs((I1 - I2)/(I2 - I3))) / np.log(2)

aitken_text = f"""
Aitken:
I1 = {I1:.6f}
I2 = {I2:.6f}
I3 = {I3:.6f}
p ≈ {p:.4f}
"""

# ----------------------------
# АДАПТИВНИЙ СІМПСОН (з лічильником)
# ----------------------------
def adaptive_simpson(f, a, b, eps, counter):
    counter[0] += 1

    def simpson_local(f, a, b):
        c = (a + b) / 2
        return (b - a)/6 * (f(a) + 4*f(c) + f(b))

    c = (a + b) / 2
    S = simpson_local(f, a, b)
    S1 = simpson_local(f, a, c)
    S2 = simpson_local(f, c, b)

    if abs(S1 + S2 - S) < 15*eps:
        return S1 + S2 + (S1 + S2 - S)/15
    else:
        return (adaptive_simpson(f, a, c, eps/2, counter) +
                adaptive_simpson(f, c, b, eps/2, counter))

# ----------------------------
# АДАПТИВНИЙ (одне значення)
# ----------------------------
eps = 0.001
counter = [0]
I_adaptive = adaptive_simpson(f, a, b, eps, counter)
error_adaptive = abs(I_exact - I_adaptive)

adaptive_text = f"""
Adaptive Simpson:
I = {I_adaptive:.6f}
error = {error_adaptive:.6f}
calls = {counter[0]}
"""

# ----------------------------
# ДОСЛІДЖЕННЯ (п.9)
# ----------------------------
eps_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
adaptive_errors = []
calls = []

for eps in eps_values:
    counter = [0]
    I = adaptive_simpson(f, a, b, eps, counter)
    adaptive_errors.append(abs(I_exact - I))
    calls.append(counter[0])

# ----------------------------
# ГРАФІКИ
# ----------------------------
fig, axs = plt.subplots(3, 2, figsize=(14, 15))

# 1. Графік функції
x = np.linspace(a, b, 1000)
axs[0, 0].plot(x, f(x))
axs[0, 0].set_title("Функція f(x)")
axs[0, 0].grid()

# 2. Похибка Сімпсона
axs[0, 1].plot(n_values, errors, marker='o')
axs[0, 1].set_title("Похибка vs n (Сімпсон)")
axs[0, 1].set_xlabel("n")
axs[0, 1].set_ylabel("error")
axs[0, 1].grid()

# 3. Таблиця + методи
axs[1, 0].axis('off')
axs[1, 0].text(0, 1, table_text + runge_text + aitken_text,
               fontsize=10, verticalalignment='top', family='monospace')

# 4. Адаптивний
axs[1, 1].axis('off')
axs[1, 1].text(0, 1,
               f"Exact ≈ {I_exact:.6f}\n" + adaptive_text,
               fontsize=12, verticalalignment='top')

# 5. Похибка від eps
axs[2, 0].plot(eps_values, adaptive_errors, marker='o')
axs[2, 0].set_xscale('log')
axs[2, 0].set_yscale('log')
axs[2, 0].set_title("Похибка адаптивного методу vs eps")
axs[2, 0].set_xlabel("eps")
axs[2, 0].set_ylabel("error")
axs[2, 0].grid()

# 6. Кількість викликів
axs[2, 1].plot(eps_values, calls, marker='o')
axs[2, 1].set_xscale('log')
axs[2, 1].set_title("Кількість обчислень vs eps")
axs[2, 1].set_xlabel("eps")
axs[2, 1].set_ylabel("calls")
axs[2, 1].grid()

plt.tight_layout()
plt.show()