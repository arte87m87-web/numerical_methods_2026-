import random
import tkinter as tk
from tkinter import scrolledtext
import os

# ----------------------------
# Генерація матриці
# ----------------------------
def generate_matrix(n):
    A = [[random.uniform(-5, 5) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        A[i][i] = sum(abs(A[i][j]) for j in range(n) if j != i) + random.uniform(1, 5)
    return A

# ----------------------------
# Запис у файл
# ----------------------------
def write_matrix(filename, A):
    with open(filename, 'w') as f:
        for row in A:
            f.write(" ".join(map(str, row)) + "\n")

def write_vector(filename, b):
    with open(filename, 'w') as f:
        f.write(" ".join(map(str, b)))

# ----------------------------
# Зчитування
# ----------------------------
def read_matrix(filename):
    with open(filename, 'r') as f:
        return [list(map(float, line.split())) for line in f]

def read_vector(filename):
    with open(filename, 'r') as f:
        return list(map(float, f.read().split()))

# ----------------------------
# Множення
# ----------------------------
def mat_vec(A, x):
    return [sum(A[i][j]*x[j] for j in range(len(A))) for i in range(len(A))]

# ----------------------------
# Норма
# ----------------------------
def norm(x):
    return max(abs(xi) for xi in x)

# ----------------------------
# Проста ітерація
# ----------------------------
def simple_iteration(A, b, x0, eps):
    n = len(A)
    tau = 1 / max(sum(abs(A[i][j]) for j in range(n)) for i in range(n))
    x = x0[:]

    for k in range(10000):
        Ax = mat_vec(A, x)
        x_new = [x[i] - tau*(Ax[i] - b[i]) for i in range(n)]

        if norm([x_new[i] - x[i] for i in range(n)]) < eps:
            return x_new, k+1

        x = x_new

    return x, 10000

# ----------------------------
# Якобі
# ----------------------------
def jacobi(A, b, x0, eps):
    n = len(A)
    x = x0[:]

    for k in range(10000):
        x_new = [0]*n
        for i in range(n):
            s = sum(A[i][j]*x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]

        if norm([x_new[i] - x[i] for i in range(n)]) < eps:
            return x_new, k+1

        x = x_new

    return x, 10000

# ----------------------------
# Зейдель
# ----------------------------
def seidel(A, b, x0, eps):
    n = len(A)
    x = x0[:]

    for k in range(10000):
        x_new = x[:]
        for i in range(n):
            s1 = sum(A[i][j]*x_new[j] for j in range(i))
            s2 = sum(A[i][j]*x[j] for j in range(i+1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]

        if norm([x_new[i] - x[i] for i in range(n)]) < eps:
            return x_new, k+1

        x = x_new

    return x, 10000

# ----------------------------
# РОЗРАХУНОК
# ----------------------------
def calculate_and_show():
    n = 100
    eps = 1e-14

    text.insert(tk.END, "Генерація матриці...\n")

    A = generate_matrix(n)
    x_true = [2.5]*n
    b = mat_vec(A, x_true)

    path = os.getcwd()

    write_matrix("A.txt", A)
    write_vector("B.txt", b)

    text.insert(tk.END, f"Файли збережені в:\n{path}\n\n")

    A = read_matrix("A.txt")
    b = read_vector("B.txt")

    x0 = [1.0]*n

    text.insert(tk.END, "Розв'язання...\n\n")

    x1, it1 = simple_iteration(A, b, x0, eps)
    text.insert(tk.END, f"Проста ітерація: {it1} ітерацій\n")

    x2, it2 = jacobi(A, b, x0, eps)
    text.insert(tk.END, f"Якобі: {it2} ітерацій\n")

    x3, it3 = seidel(A, b, x0, eps)
    text.insert(tk.END, f"Зейдель: {it3} ітерацій\n")

    # ----------------------------
    # Вивід значень
    # ----------------------------
    text.insert(tk.END, "\nПерші 5 значень:\n")

    for i in range(5):
        text.insert(tk.END,
            f"x[{i}] | Ітерація: {x1[i]:.10f} | Якобі: {x2[i]:.10f} | Зейдель: {x3[i]:.10f}\n")

    # ----------------------------
    # Різниця між методами
    # ----------------------------
    text.insert(tk.END, "\nРізниця між методами:\n")

    for i in range(5):
        d12 = abs(x1[i] - x2[i])
        d13 = abs(x1[i] - x3[i])
        d23 = abs(x2[i] - x3[i])

        text.insert(tk.END,
            f"x[{i}] | |Іт-Як|={d12:.2e} | |Іт-Зей|={d13:.2e} | |Як-Зей|={d23:.2e}\n")

# ----------------------------
# GUI
# ----------------------------
root = tk.Tk()
root.title("Лабораторна №8")

text = scrolledtext.ScrolledText(root, width=90, height=30)
text.pack()

root.after(100, calculate_and_show)

root.mainloop()