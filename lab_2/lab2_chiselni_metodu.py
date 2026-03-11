import csv
import numpy as np
import matplotlib.pyplot as plt

# --- Зчитування даних ---
def read_data(filename):
    x, y = [], []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['n']))
            y.append(float(row['t']))
    return np.array(x), np.array(y)

# --- Таблиця розділених різниць ---
def divided_differences(x, y):
    n = len(x)
    table = np.zeros((n, n))
    table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i])
    return table

# --- Поліном Ньютона ---
def newton_polynomial(x, table, value):
    n = len(x)
    result = table[0][0]
    product = 1
    for i in range(1, n):
        product *= (value - x[i - 1])
        result += table[0][i] * product
    return result

# --- Основна програма ---

# Зчитуємо дані з CSV
x, y = read_data(r"C:\Users\Nik\Desktop\1 курс\Чисельні методи\Lab2_chm\data.csv")

# Будуємо таблицю розділених різниць для початкових даних
table = divided_differences(x, y)

# Прогноз для n=6000 на основі всіх вузлів
value = 6000
forecast_full = newton_polynomial(x, table, value)
print(f"Прогноз часу виконання при n={value} (всі вузли): {forecast_full:.5f}")

# Малюємо графік для всіх вузлів
x_plot = np.linspace(min(x), max(x), 200)
y_plot_full = [newton_polynomial(x, table, val) for val in x_plot]

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='red', label="Експериментальні точки")
plt.plot(x_plot, y_plot_full, label="Поліном Ньютона (всі вузли)")
plt.xlabel("n")
plt.ylabel("t")
plt.title("Інтерполяція многочленом Ньютона (всі вузли)")
plt.legend()
plt.grid()
plt.show()

# --- Прогнози та графіки для 5, 10, 20 вузлів ---

nodes_list = [5, 10, 20]
errors = []
true_value = forecast_full

for nodes in nodes_list:
    # Вибираємо рівномірно розподілені вузли у діапазоні
    x_sub = np.linspace(min(x), max(x), nodes)
    # Значення y_sub отримуємо, інтерполюючи початкові дані многочленом Ньютона
    y_sub = [newton_polynomial(x, table, val) for val in x_sub]

    # Таблиця розділених різниць для підмножини
    table_sub = divided_differences(x_sub, y_sub)

    # Прогноз для n=6000 на підмножині вузлів
    forecast_sub = newton_polynomial(x_sub, table_sub, value)
    error = abs(true_value - forecast_sub)
    errors.append(error)

    print(f"\nКількість вузлів: {nodes}")
    print(f"Прогноз: {forecast_sub:.5f}")
    print(f"Похибка: {error:.5f}")

    # Побудова графіка інтерполяції для цієї кількості вузлів
    y_plot_sub = [newton_polynomial(x_sub, table_sub, val) for val in x_plot]

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='black', label="Експериментальні точки")
    plt.plot(x_plot, y_plot_sub, label=f"Інтерполяція Ньютона ({nodes} вузлів)")
    plt.xlabel("n")
    plt.ylabel("t")
    plt.title(f"Інтерполяція для {nodes} вузлів")
    plt.legend()
    plt.grid()
    plt.show()

# --- Графік похибок прогнозу залежно від кількості вузлів ---

plt.figure(figsize=(8, 5))
plt.plot(nodes_list, errors, marker='o', linestyle='-', color='blue')
plt.xlabel("Кількість вузлів")
plt.ylabel("Похибка прогнозу")
plt.title("Вплив кількості вузлів на точність прогнозу")
plt.grid()
plt.show()

# --- Порівняння інтерполяційних кривих для 5, 10, 20 вузлів на одному графіку ---

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='black', label="Експериментальні точки")

for nodes in nodes_list:
    x_sub = np.linspace(min(x), max(x), nodes)
    y_sub = [newton_polynomial(x, table, val) for val in x_sub]
    table_sub = divided_differences(x_sub, y_sub)
    y_plot_sub = [newton_polynomial(x_sub, table_sub, val) for val in x_plot]
    plt.plot(x_plot, y_plot_sub, label=f"{nodes} вузлів")

plt.xlabel("n")
plt.ylabel("t")
plt.title("Порівняння інтерполяції для різної кількості вузлів")
plt.legend()
plt.grid()
plt.show()