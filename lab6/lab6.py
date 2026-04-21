import numpy as np
import matplotlib.pyplot as plt

# =========================
# LU-розклад
# =========================
def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        U[i][i] = 1

    for k in range(n):
        for i in range(k, n):
            L[i][k] = A[i][k] - sum(L[i][j] * U[j][k] for j in range(k))

        for j in range(k + 1, n):
            U[k][j] = (A[k][j] - sum(L[k][m] * U[m][j] for m in range(k))) / L[k][k]

    return L, U


def forward_substitution(L, B):
    n = len(B)
    Z = np.zeros(n)

    for i in range(n):
        Z[i] = (B[i] - sum(L[i][j] * Z[j] for j in range(i))) / L[i][i]

    return Z


def backward_substitution(U, Z):
    n = len(Z)
    X = np.zeros(n)

    for i in reversed(range(n)):
        X[i] = Z[i] - sum(U[i][j] * X[j] for j in range(i + 1, n))

    return X


def iterative_refinement(A, B, X0, L, U, iterations=10):
    X = X0.copy()
    errors = []

    for _ in range(iterations):
        R = B - A @ X
        Z = forward_substitution(L, R)
        dX = backward_substitution(U, Z)

        X = X + dX
        errors.append(np.linalg.norm(dX))

    return X, errors


# =========================
# Дані
# =========================
A = np.array([
    [4, -1, 1],
    [-1, 4, -1],
    [1, -1, 4]
], dtype=float)

B = np.array([6, 6, 6], dtype=float)

# =========================
# Обчислення
# =========================
L, U = lu_decomposition(A)
Z = forward_substitution(L, B)
X = backward_substitution(U, Z)
X_refined, errors = iterative_refinement(A, B, X, L, U)

# =========================
# Формування тексту
# =========================
def matrix_to_str(M, name):
    return f"{name}:\n" + "\n".join(
        [" ".join([f"{val:8.3f}" for val in row]) for row in M]
    )

text_output = (
    matrix_to_str(A, "A") + "\n\n" +
    matrix_to_str(L, "L") + "\n\n" +
    matrix_to_str(U, "U") + "\n\n" +
    f"X (LU): {np.round(X, 4)}\n\n" +
    f"X (уточнений): {np.round(X_refined, 4)}"
)

# =========================
# ВІЗУАЛІЗАЦІЯ (ВСЕ В ОДНОМУ ВІКНІ)
# =========================
fig = plt.figure(figsize=(14, 8))

# Текстова панель
ax_text = plt.subplot2grid((2, 2), (0, 0), colspan=2)
ax_text.axis('off')
ax_text.text(0, 1, text_output, fontsize=10, va='top', family='monospace')

# Графік збіжності
ax1 = plt.subplot2grid((2, 2), (1, 0))
ax1.plot(errors, marker='o')
ax1.set_title("Збіжність ітерацій")
ax1.set_xlabel("Ітерація")
ax1.set_ylabel("Похибка")
ax1.grid()

# Порівняння розв’язків
ax2 = plt.subplot2grid((2, 2), (1, 1))
ax2.plot(X, 'o-', label='Початковий')
ax2.plot(X_refined, 's--', label='Уточнений')
ax2.set_title("Порівняння розв’язків")
ax2.legend()
ax2.grid()

plt.tight_layout()
plt.show()