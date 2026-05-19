"""
Лабораторна робота: Метод Хука-Джівса для розв'язання
систем нелінійних рівнянь
"""

import math
import os
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────
# ВІКНО ДЛЯ ВИВОДУ
# ─────────────────────────────────────────────────

root = tk.Tk()
root.title("Метод Хука-Джівса")
root.geometry("950x720")

text_area = ScrolledText(
    root,
    wrap=tk.WORD,
    font=("Consolas", 11)
)

text_area.pack(expand=True, fill="both")


def log(message=""):
    """Вивід тексту у вікно."""
    text_area.insert(tk.END, str(message) + "\n")
    text_area.see(tk.END)
    root.update()


# ─────────────────────────────────────────────────
# Папка для результатів
# ─────────────────────────────────────────────────

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def out(filename):
    return os.path.join(OUTPUT_DIR, filename)


# ─────────────────────────────────────────────────
# 1. Система нелінійних рівнянь
# ─────────────────────────────────────────────────

def f1(x1, x2):
    return x1**2 + x2**2 - 4


def f2(x1, x2):
    return x1 * x2 - 1


def Phi(X):
    x1, x2 = X
    return f1(x1, x2)**2 + f2(x1, x2)**2


# ─────────────────────────────────────────────────
# 2. Метод Хука-Джівса
# ─────────────────────────────────────────────────

def hooke_jeeves(
    func,
    X0,
    delta,
    q=0.5,
    p=2.0,
    eps1=1e-4,
    eps2=1e-6,
    max_iter=10000
):

    n = len(X0)

    X_base = list(X0)
    step = list(delta)

    trajectory = [list(X_base)]
    total_steps = 0

    for iteration in range(max_iter):

        # ── Дослідницький пошук ───────────────────
        X_new = list(X_base)

        for i in range(n):

            # Позитивний напрямок
            X_try = list(X_new)
            X_try[i] += step[i]

            if func(X_try) < func(X_new):
                X_new = X_try

            else:
                # Негативний напрямок
                X_try = list(X_new)
                X_try[i] -= step[i]

                if func(X_try) < func(X_new):
                    X_new = X_try

        total_steps += 1
        trajectory.append(list(X_new))

        # ── Успіх пошуку ──────────────────────────
        if func(X_new) < func(X_base):

            # Хід за зразком
            X_pattern = [
                X_new[i] + p * (X_new[i] - X_base[i])
                for i in range(n)
            ]

            X_base = list(X_new)

            # Повторний пошук
            X_explore = list(X_pattern)

            for i in range(n):

                X_try = list(X_explore)
                X_try[i] += step[i]

                if func(X_try) < func(X_explore):
                    X_explore = X_try

                else:
                    X_try = list(X_explore)
                    X_try[i] -= step[i]

                    if func(X_try) < func(X_explore):
                        X_explore = X_try

            total_steps += 1
            trajectory.append(list(X_explore))

            if func(X_explore) < func(X_base):
                X_base = list(X_explore)

        else:
            # Зменшення кроку
            step = [s * q for s in step]

        # ── Критерії зупинки ──────────────────────
        step_norm = math.sqrt(sum(s**2 for s in step))

        if step_norm < eps1 and func(X_base) < eps2:

            log(
                f"  Зупинка: "
                f"крок = {step_norm:.2e} < eps1={eps1}, "
                f"Φ = {func(X_base):.2e} < eps2={eps2}"
            )

            break

        if step_norm < eps1 * 0.01:

            log(f"  Зупинка: крок дуже малий ({step_norm:.2e})")

            break

    return X_base, trajectory, total_steps


# ─────────────────────────────────────────────────
# 3. Побудова графіків системи
# ─────────────────────────────────────────────────

def plot_system():

    x1 = np.linspace(-3, 3, 400)
    x2 = np.linspace(-3, 3, 400)

    X1, X2 = np.meshgrid(x1, x2)

    F1 = X1**2 + X2**2 - 4
    F2 = X1 * X2 - 1

    plt.figure(figsize=(7, 6))

    c1 = plt.contour(
        X1,
        X2,
        F1,
        levels=[0],
        colors='blue'
    )

    c2 = plt.contour(
        X1,
        X2,
        F2,
        levels=[0],
        colors='red'
    )

    plt.clabel(c1, fmt='f1=0')
    plt.clabel(c2, fmt='f2=0')

    # Точні розв'язки
    for sign in [1, -1]:

        x1s_sq = (4 + sign * math.sqrt(12)) / 2

        if x1s_sq > 0:

            for s2 in [1, -1]:

                x1s = s2 * math.sqrt(x1s_sq)
                x2s = 1 / x1s

                plt.plot(
                    x1s,
                    x2s,
                    'k*',
                    markersize=12,
                    label=f'({x1s:.3f}, {x2s:.3f})'
                )

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    plt.grid(True, alpha=0.3)

    plt.xlabel('x₁')
    plt.ylabel('x₂')

    plt.title(
        'Система нелінійних рівнянь\n'
        '(синя: x₁²+x₂²=4, червона: x₁x₂=1)'
    )

    plt.legend(title='Розв\'язки')

    plt.tight_layout()

    filename = out('system_plot.png')

    plt.savefig(filename, dpi=120)
    plt.close()

    log(f"  Графік системи збережено: {filename}")


# ─────────────────────────────────────────────────
# 4. Тест Розенброка
# ─────────────────────────────────────────────────

def run_test():

    log("\n" + "=" * 60)
    log("ТЕСТ методу Хука-Джівса на функції Розенброка")
    log("  f(x1,x2) = (1-x1)^2 + 100*(x2-x1^2)^2")
    log("  Мінімум: (1, 1), f_min = 0")
    log("=" * 60)

    def rosenbrock(X):

        x1, x2 = X

        return (
            (1 - x1)**2 +
            100 * (x2 - x1**2)**2
        )

    X0 = [-1.0, 1.0]
    delta = [0.5, 0.5]

    X_min, traj, steps = hooke_jeeves(
        rosenbrock,
        X0,
        delta,
        q=0.5,
        p=2.0,
        eps1=1e-6,
        eps2=1e-10
    )

    log(f"  X0          = {X0}")
    log(f"  Результат   = [{X_min[0]:.8f}, {X_min[1]:.8f}]")
    log(f"  f(X_min)    = {rosenbrock(X_min):.2e}")
    log(f"  Кроки       = {steps}")


# ─────────────────────────────────────────────────
# 5. Основна задача
# ─────────────────────────────────────────────────

def run_main():

    # ── Пункт 1 ──────────────────────────────────
    log("\n" + "=" * 60)
    log("ПУНКТ 1: Побудова графіків системи рівнянь")
    log("=" * 60)

    plot_system()

    # ── Пункт 2 & 4 ──────────────────────────────
    log("\n" + "=" * 60)
    log("ПУНКТИ 2 & 4: Метод Хука-Джівса")
    log("  f1(x1,x2) = x1² + x2² - 4 = 0")
    log("  f2(x1,x2) = x1·x2 - 1     = 0")
    log("  Φ(X) = f1² + f2²")
    log("=" * 60)

    # Параметри
    X0 = [1.5, 0.5]
    delta = [0.5, 0.5]

    q = 0.5
    p = 2.0

    eps1 = 1e-6
    eps2 = 1e-10

    log(f"\n  X⁽⁰⁾ = {X0}")
    log(f"  ΔX   = {delta}")
    log(f"  q={q}, p={p}, ε₁={eps1}, ε₂={eps2}")

    X_sol, trajectory, steps = hooke_jeeves(
        Phi,
        X0,
        delta,
        q=q,
        p=p,
        eps1=eps1,
        eps2=eps2
    )

    log(f"\n  ── Результат ──────────────────────────")
    log(f"  X*     = [{X_sol[0]:.8f}, {X_sol[1]:.8f}]")
    log(f"  Φ(X*)  = {Phi(X_sol):.4e}")
    log(f"  f1(X*) = {f1(*X_sol):.4e}")
    log(f"  f2(X*) = {f2(*X_sol):.4e}")
    log(f"  Кроки  = {steps}")

    # ── Збереження траєкторії ────────────────────
    log("\n" + "=" * 60)
    log("ПУНКТ 5: Збереження траєкторії")
    log("=" * 60)

    traj_file = out('trajectory.txt')

    with open(traj_file, 'w', encoding='utf-8') as f:

        f.write("Крок\tx1\t\t\tx2\t\t\tΦ(X)\n")
        f.write("-" * 65 + "\n")

        for i, point in enumerate(trajectory):

            phi_val = Phi(point)

            f.write(
                f"{i}\t"
                f"{point[0]: .10f}\t"
                f"{point[1]: .10f}\t"
                f"{phi_val:.6e}\n"
            )

        f.write("-" * 65 + "\n")
        f.write(f"Всього кроків: {steps}\n")

        f.write(
            f"Розв'язок: "
            f"x1={X_sol[0]:.8f}, "
            f"x2={X_sol[1]:.8f}\n"
        )

        f.write(f"Φ(X*) = {Phi(X_sol):.4e}\n")

    log(f"  Файл збережено: {traj_file}")

    # ── Графік траєкторії ────────────────────────
    x1 = np.linspace(-0.5, 2.5, 400)
    x2 = np.linspace(-0.5, 2.5, 400)

    X1, X2 = np.meshgrid(x1, x2)

    Z = (
        (X1**2 + X2**2 - 4)**2 +
        (X1 * X2 - 1)**2
    )

    traj_arr = np.array(trajectory)

    plt.figure(figsize=(8, 6))

    cp = plt.contourf(
        X1,
        X2,
        np.log1p(Z),
        levels=40,
        cmap='viridis'
    )

    plt.colorbar(cp, label='log(1 + Φ)')

    plt.contour(
        X1,
        X2,
        Z,
        levels=[0],
        colors='white',
        linewidths=2
    )

    plt.plot(
        traj_arr[:, 0],
        traj_arr[:, 1],
        'w.-',
        linewidth=0.8,
        markersize=3,
        alpha=0.7,
        label='Траєкторія'
    )

    plt.plot(
        X0[0],
        X0[1],
        'go',
        markersize=10,
        label=f'Старт {X0}'
    )

    plt.plot(
        X_sol[0],
        X_sol[1],
        'r*',
        markersize=14,
        label=f'Мінімум ({X_sol[0]:.4f}, {X_sol[1]:.4f})'
    )

    plt.xlabel('x₁')
    plt.ylabel('x₂')

    plt.title(
        f'Траєкторія методу Хука-Джівса\n'
        f'(кроків: {steps})'
    )

    plt.legend()

    plt.tight_layout()

    filename = out('trajectory_plot.png')

    plt.savefig(filename, dpi=120)
    plt.close()

    log(f"  Графік траєкторії збережено: {filename}")

    return X_sol, steps


# ─────────────────────────────────────────────────
# ГОЛОВНА ПРОГРАМА
# ─────────────────────────────────────────────────

if __name__ == "__main__":

    log("=" * 60)
    log("  МЕТОД ХУКА-ДЖІВСА")
    log("  СИСТЕМА НЕЛІНІЙНИХ РІВНЯНЬ")
    log("=" * 60)

    # Тест
    run_test()

    # Основна задача
    X_sol, steps = run_main()

    log("\n" + "=" * 60)
    log("  ПІДСУМОК")
    log("=" * 60)

    log(
        f"  Розв'язок системи: "
        f"x1 ≈ {X_sol[0]:.6f}, "
        f"x2 ≈ {X_sol[1]:.6f}"
    )

    log(
        f"  Перевірка: "
        f"f1 = {f1(*X_sol):.2e}, "
        f"f2 = {f2(*X_sol):.2e}"
    )

    log(f"  Кількість кроків: {steps}")

    log("\n  Збережені файли:")
    log("    system_plot.png")
    log("    trajectory_plot.png")
    log("    trajectory.txt")

    log("=" * 60)

    # Запуск GUI
    root.mainloop()