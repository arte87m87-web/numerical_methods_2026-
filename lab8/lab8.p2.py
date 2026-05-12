import math
import tkinter as tk
from tkinter import ttk

# --- 1. ТАБУЛЯЦІЯ ТА ФАЙЛИ ---

def tabulate_function(a, b, h):
    results = []
    try:
        with open("tabulation.txt", "w", encoding="utf-8") as f:
            f.write("x\t\tf(x)\n")
            x = a
            while x <= b + 0.0001:
                y = F(x)
                results.append((x, y))
                f.write(f"{x:.4f}\t{y:.6f}\n")
                x += h
        return results
    except Exception as e:
        print(f"Помилка запису у файл: {e}")
        return []

def load_coeffs_from_file(filename="coeffs.txt"):
    try:
        with open(filename, "r") as f:
            line = f.read().replace(',', ' ')
            return [float(c) for c in line.split()]
    except:
        # Якщо файлу немає, повертаємо коефіцієнти для x^3 - 2x^2 - 5x + 6
        return [10.0, 5.0, 2.0, 1.0]

# --- 2. МАТЕМАТИЧНІ ФУНКЦІЇ ---

def F(x):
    return math.sin(x) - 0.5 * x

def dF(x):
    return math.cos(x) - 0.5

def ddF(x):
    return -math.sin(x)

def phi(x):
    return 2 * math.sin(x)

# --- 3. ВСІ ІТЕРАЦІЙНІ МЕТОДИ ---

def simple_iteration(x0, eps):
    it = 0
    xn = x0
    while it < 500:
        x_next = phi(xn)
        if abs(x_next - xn) < eps:
            return x_next, it
        xn = x_next
        it += 1
    return xn, it

def newton_method(x0, eps):
    it = 0
    xn = x0
    while it < 500:
        f_val = F(xn)
        df_val = dF(xn)
        x_next = xn - f_val / df_val
        if abs(x_next - xn) < eps and abs(f_val) < eps:
            return x_next, it
        xn = x_next
        it += 1
    return xn, it

def chebyshev_method(x0, eps):
    it = 0
    xn = x0
    while it < 500:
        f, df, ddf = F(xn), dF(xn), ddF(xn)
        x_next = xn - f/df - 0.5 * (f**2 * ddf) / (df**3)
        if abs(x_next - xn) < eps:
            return x_next, it
        xn = x_next
        it += 1
    return xn, it

def chord_method(x_prev, xn, eps):
    it = 0
    while it < 500:
        f_xn = F(xn)
        f_prev = F(x_prev)
        x_next = xn - f_xn * (xn - x_prev) / (f_xn - f_prev)
        if abs(x_next - xn) < eps:
            return x_next, it
        x_prev, xn = xn, x_next
        it += 1
    return xn, it

def parabola_method(x0, x1, x2, eps):
    it = 0
    while it < 100:
        f0, f1, f2 = F(x0), F(x1), F(x2)
        h1, h2 = x1 - x0, x2 - x1
        if h1 == 0 or h2 == 0: break
        d1, d2 = (f1 - f0) / h1, (f2 - f1) / h2
        d = (d2 - d1) / (h2 + h1)
        b = d2 + h2 * d
        try:
            D = math.sqrt(b**2 - 4 * f2 * d)
        except:
            D = 0 
        den = b + D if abs(b + D) > abs(b - D) else b - D
        if den == 0: break
        dx = -2 * f2 / den
        x3 = x2 + dx
        if abs(dx) < eps: return x3, it
        x0, x1, x2 = x1, x2, x3
        it += 1
    return x2, it

def inverse_interpolation(x0, x1, x2, eps):
    it = 0
    p = [(x0, F(x0)), (x1, F(x1)), (x2, F(x2))]
    while it < 100:
        x_next = 0
        for i in range(3):
            li = 1
            for j in range(3):
                if i != j:
                    if (p[i][1] - p[j][1]) == 0: continue
                    li *= (0 - p[j][1]) / (p[i][1] - p[j][1])
            x_next += p[i][0] * li
        if abs(x_next - p[2][0]) < eps: return x_next, it
        p.pop(0)
        p.append((x_next, F(x_next)))
        it += 1
    return p[-1][0], it

# --- 4. СХЕМА ГОРНЕРА ---

def horner_eval(coeffs, x):
    res = coeffs[-1]
    der = 0
    for i in range(len(coeffs)-2, -1, -1):
        der = res + x * der
        res = coeffs[i] + x * res
    return res, der

def newton_horner(coeffs, x0, eps):
    it = 0
    xn = x0
    while it < 100:
        f, df = horner_eval(coeffs, xn)
        if abs(df) < 1e-10: break
        x_next = xn - f / df
        if abs(x_next - xn) < eps: return x_next, it
        xn = x_next
        it += 1
    return xn, it

# --- НОВЕ: МЕТОД ЛІНА ДЛЯ КОМПЛЕКСНИХ КОРЕНІВ ---

def lin_method(coeffs, p0, q0, eps):
    """
    Метод Ліна для виділення квадратного тричлена x^2 + px + q.
    coeffs: [a0, a1, a2, a3] (від нижчого степеня до вищого)
    """
    it = 0
    p, q = p0, q0
    n = len(coeffs) - 1
    
    while it < 100:
        b = [0] * (n + 1)
        b[n] = coeffs[n]
        b[n-1] = coeffs[n-1] - p * b[n]
        for i in range(n-2, -1, -1):
            b[i] = coeffs[i] - p * b[i+1] - q * b[i+2]
        
        # Обчислення нових наближень p та q
        p_new = (coeffs[1] - b[1]) / b[2] if abs(b[2]) > 1e-10 else p
        q_new = coeffs[0] / b[2] if abs(b[2]) > 1e-10 else q
        
        if abs(p_new - p) < eps and abs(q_new - q) < eps:
            # Розв'язок x^2 + px + q = 0
            discr = p**2 - 4*q
            if discr >= 0:
                r1 = (-p + math.sqrt(discr)) / 2
                r2 = (-p - math.sqrt(discr)) / 2
                return (complex(r1, 0), complex(r2, 0)), it
            else:
                real_part = -p / 2
                imag_part = math.sqrt(-discr) / 2
                return (complex(real_part, imag_part), complex(real_part, -imag_part)), it
        
        p, q = p_new, q_new
        it += 1
    return None, it

# --- 5. ДОДАТОК (GUI) ---

class LabApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Лабораторна робота №8 - ЛНУ")
        self.root.geometry("800x650")
        
        ttk.Label(root, text="Результати чисельних методів", font=("Arial", 14, "bold")).pack(pady=10)

        self.tree = ttk.Treeview(root, columns=("Method", "Root", "Iterations"), show='headings', height=10)
        self.tree.heading("Method", text="Метод")
        self.tree.heading("Root", text="Корінь")
        self.tree.heading("Iterations", text="Ітерації")
        self.tree.pack(pady=10, padx=20, fill="x")

        btn_frame = ttk.Frame(root)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="1. Табуляція", command=self.run_tabulation).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="2. Розрахувати", command=self.run_methods).pack(side="left", padx=5)

        self.info_text = tk.Text(root, height=12, font=("Consolas", 10))
        self.info_text.pack(pady=10, padx=20, fill="both")

    def run_tabulation(self):
        res = tabulate_function(0, 4, 0.2)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, ">>> Пункт 1: Табуляція збережена в 'tabulation.txt'\n")
        self.info_text.insert(tk.END, "Перші 5 значень:\n")
        for x, y in res[:5]:
            self.info_text.insert(tk.END, f"x: {x:.2f} -> f(x): {y:.4f}\n")

    def run_methods(self):
        eps = 1e-5
        for i in self.tree.get_children(): self.tree.delete(i)
        
        # Розрахунки для трансцендентної функції
        results = [
            ("Проста ітерація", simple_iteration(1.8, eps)),
            ("Метод Ньютона", newton_method(1.8, eps)),
            ("Метод Чебишева", chebyshev_method(1.8, eps)),
            ("Метод хорд", chord_method(1.5, 2.5, eps)),
            ("Метод парабол", parabola_method(1.5, 1.8, 2.0, eps)),
            ("Зворотна інтерпол.", inverse_interpolation(1.5, 1.8, 2.0, eps))
        ]

        for name, (res, iters) in results:
            self.tree.insert("", tk.END, values=(name, f"{res:.6f}", iters))

        # Алгебраїчне рівняння
        coeffs = load_coeffs_from_file()
        alg_root, alg_iters = newton_horner(coeffs, 2.0, eps)

        self.info_text.insert(tk.END, "\n--- Пункт 8: Алгебраїчне рівняння (Схема Горнера) ---\n")
        self.info_text.insert(tk.END, f"Коефіцієнти: {coeffs}\n")
        self.info_text.insert(tk.END, f"Знайдений дійсний корінь: {alg_root:.6f}\n")
        self.info_text.insert(tk.END, f"Кількість ітерацій: {alg_iters}\n")

        # Нове: Пункт 9 - Метод Ліна
        self.info_text.insert(tk.END, "\n--- Пункт 9: Комплексні корені (Метод Ліна) ---\n")
        # Початкові наближення для p та q
        complex_roots, lin_iters = lin_method(coeffs, 1.0, 1.0, eps)
        if complex_roots:
            self.info_text.insert(tk.END, f"Корінь 1: {complex_roots[0]}\n")
            self.info_text.insert(tk.END, f"Корінь 2: {complex_roots[1]}\n")
            self.info_text.insert(tk.END, f"Кількість ітерацій: {lin_iters}\n")
        else:
            self.info_text.insert(tk.END, "Метод Ліна не збігся.\n")

if __name__ == "__main__":
    try:
        # Приклад для рівняння x^3 - 2x^2 - 5x + 6 (має корені 1, 3, -2 - всі дійсні)
        # Або для комплексних коренів можна змінити на інше
        with open("coeffs.txt", "w") as f: f.write("10 5 2 1")
    except: pass
    
    root = tk.Tk()
    app = LabApp(root)
    root.mainloop()