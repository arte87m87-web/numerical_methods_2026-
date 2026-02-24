import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1. Вхідні дані (координати + висота)
# ---------------------------

results = [
    {"latitude":48.164214,"longitude":24.536044,"elevation":1264.0},
    {"latitude":48.164983,"longitude":24.534836,"elevation":1285.0},
    {"latitude":48.165605,"longitude":24.534068,"elevation":1285.0},
    {"latitude":48.166228,"longitude":24.532915,"elevation":1333.0},
    {"latitude":48.166777,"longitude":24.531927,"elevation":1310.0},
    {"latitude":48.167326,"longitude":24.530884,"elevation":1318.0},
    {"latitude":48.167011,"longitude":24.530061,"elevation":1318.0},
    {"latitude":48.166053,"longitude":24.528039,"elevation":1339.0},
    {"latitude":48.166655,"longitude":24.526064,"elevation":1375.0},
    {"latitude":48.166497,"longitude":24.523574,"elevation":1417.0},
    {"latitude":48.166128,"longitude":24.520214,"elevation":1486.0},
    {"latitude":48.165416,"longitude":24.517170,"elevation":1524.0},
    {"latitude":48.164546,"longitude":24.514640,"elevation":1553.0},
    {"latitude":48.163412,"longitude":24.512980,"elevation":1630.0},
    {"latitude":48.162331,"longitude":24.511715,"elevation":1757.0},
    {"latitude":48.162015,"longitude":24.509462,"elevation":1794.0},
    {"latitude":48.162147,"longitude":24.506932,"elevation":1828.0},
    {"latitude":48.161751,"longitude":24.504244,"elevation":1887.0},
    {"latitude":48.161197,"longitude":24.501793,"elevation":1975.0},
    {"latitude":48.160580,"longitude":24.500537,"elevation":1975.0},
    {"latitude":48.160250,"longitude":24.500106,"elevation":2031.0}
]

# ---------------------------
# 2. Табуляція вхідних даних
# ---------------------------

coords = [(p["latitude"], p["longitude"]) for p in results]
elev = np.array([p["elevation"] for p in results])
n = len(elev)

print("Кількість вузлів:", n)
print("№ | Latitude | Longitude | Elevation")
for i, p in enumerate(results):
    print(f"{i:2d} | {p['latitude']:.6f} | {p['longitude']:.6f} | {p['elevation']:.2f}")

# --------------------------------------
# 3. Обчислення кумулятивної відстані
# --------------------------------------

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # радіус Землі в метрах
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

dist = [0]
for i in range(1, n):
    d = haversine(*coords[i - 1], *coords[i])
    dist.append(dist[-1] + d)
dist = np.array(dist)

# --------------------------------------------------------
# 4. Метод прогонки (розв'язок трьохдіагональної системи)
# --------------------------------------------------------

def thomas(a, b, c, d):
    n = len(d)
    cp = np.zeros(n-1)
    dp = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n-1):
        denom = b[i] - a[i-1] * cp[i-1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i-1] * dp[i-1]) / denom
    dp[-1] = (d[-1] - a[-1] * dp[-2]) / (b[-1] - a[-1] * cp[-2])
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
    return x

# ----------------------------------------------------
# 5. Формування системи для знаходження коефіцієнтів c
# ----------------------------------------------------

h = np.diff(dist)
a_diag = h[1:]
b_diag = 2 * (h[:-1] + h[1:])
c_diag = h[1:]
d_vec = 3 * ((elev[2:] - elev[1:-1]) / h[1:] - (elev[1:-1] - elev[:-2]) / h[:-1])
c_coeff = np.concatenate(([0], thomas(a_diag, b_diag, c_diag, d_vec), [0]))

# ----------------------------------------------------------------------------
# 6. Знаходимо коефіцієнти a, b, c, d для кожного інтервалу сплайна
# ----------------------------------------------------------------------------

a_s = elev[:-1]
b_s = (elev[1:] - elev[:-1]) / h - h * (2*c_coeff[:-1] + c_coeff[1:]) / 3
d_s = (c_coeff[1:] - c_coeff[:-1]) / (3 * h)

# -----------------------------------
# 7. Друк коефіцієнтів у консолі
# -----------------------------------

print("\nКоефіцієнти сплайна:")
for i in range(n-1):
    print(f"Інтервал {i}: a={a_s[i]:.3f}, b={b_s[i]:.3f}, c={c_coeff[i]:.3f}, d={d_s[i]:.6f}")

# -------------------------------------------------
# 8. Побудова графіка сплайна на всьому маршруті
# -------------------------------------------------

xx = np.linspace(dist[0], dist[-1], 5000)
yy = np.zeros_like(xx)

for i in range(n - 1):
    mask = (xx >= dist[i]) & (xx <= dist[i + 1])
    dx = xx[mask] - dist[i]
    yy[mask] = a_s[i] + b_s[i] * dx + c_coeff[i] * dx**2 + d_s[i] * dx**3

plt.figure(figsize=(10, 5))
plt.plot(dist, elev, 'o', label='Вузли')
plt.plot(xx, yy, label='Кубічний сплайн')
plt.xlabel("Відстань (м)")
plt.ylabel("Висота (м)")
plt.grid(True)
plt.legend()
plt.show()