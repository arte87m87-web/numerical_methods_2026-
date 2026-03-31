import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. ФУНКЦІЯ І ПОХІДНА
# =========================

def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def dM_exact(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

def derivative(t, h):
    return (M(t + h) - M(t - h)) / (2 * h)

t0 = 1
exact = dM_exact(t0)

# =========================
# 2. ДОСЛІДЖЕННЯ ПОХИБКИ
# =========================

hs = np.logspace(-6, -1, 30)   # розширили діапазон!
errors = []

for h in hs:
    d = derivative(t0, h)
    errors.append(abs(d - exact))

errors = np.array(errors)

# =========================
# 2.1 ОПТИМАЛЬНИЙ КРОК (завдання 2)
# =========================

min_index = np.argmin(errors)
h_opt = hs[min_index]
R_opt = errors[min_index]
D_opt = derivative(t0, h_opt)

# =========================
# 3. ОСНОВНІ ОБЧИСЛЕННЯ
# =========================

h0 = 1e-3

D_h = derivative(t0, h0)
R0 = abs(D_h - exact)

h2 = 2 * h0
D_h2 = derivative(t0, h2)
R1 = abs(D_h2 - exact)   # ВИПРАВЛЕНО!

# =========================
# 4. РУНГЕ-РОМБЕРГ
# =========================

D_rr = D_h + (D_h - D_h2) / 3
R2 = abs(D_rr - exact)

# =========================
# 5. ЕЙТКЕН
# =========================

h4 = 4 * h0
D_h4 = derivative(t0, h4)

D_aitken = (D_h2**2 - D_h4 * D_h) / (2 * D_h2 - D_h4 - D_h)
R3 = abs(D_aitken - exact)

p = (1 / np.log(2)) * np.log(abs((D_h4 - D_h2) / (D_h2 - D_h)))

# =========================
# 6. ВІДОБРАЖЕННЯ
# =========================

fig, (ax_text, ax_plot) = plt.subplots(1, 2, figsize=(14, 6))

# --- ЛІВА ЧАСТИНА (ТЕКСТ) ---
ax_text.axis('off')

text = f"""
=== РЕЗУЛЬТАТИ ===

Точне значення y'(1) = {exact:.10f}

1) При h = 0.001:
D(h) = {D_h:.10f}
Похибка R0 = {R0:.15e}

2) При h і 2h:
D(h) = {D_h:.10f}
D(2h) = {D_h2:.10f}
Похибка R1 = {R1:.15e}

3) Метод Рунге-Ромберга:
D* = {D_rr:.10f}
Похибка R2 = {R2:.15e}

4) Метод Ейткена:
D* = {D_aitken:.10f}
Похибка R3 = {R3:.15e}

Порядок точності p ≈ {p:.4f}

-------------------------------
Похибки чисельного диференціювання
Оптимальний крок h₀ = {h_opt:.2e}
D(h₀) = {D_opt:.10f}
Мінімальна похибка R₀ = {R_opt:.15e}
"""

ax_text.text(0, 1, text, fontsize=11, va='top', family='monospace')

# --- ПРАВА ЧАСТИНА (ГРАФІК) ---
ax_plot.plot(hs, errors, marker='o')
ax_plot.set_xscale('log')
ax_plot.set_yscale('log')

ax_plot.set_xlabel("Крок h")
ax_plot.set_ylabel("Похибка")
ax_plot.set_title("Залежність похибки R(h)")
ax_plot.grid()

# відмітка оптимального h
ax_plot.scatter(h_opt, R_opt)
ax_plot.annotate("h₀",
                 (h_opt, R_opt),
                 textcoords="offset points",
                 xytext=(10,10))

plt.tight_layout()
plt.show()