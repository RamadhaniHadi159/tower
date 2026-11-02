import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(
    "/data/mahasiswa/romadhon/perez1/ghi//combined/GHI_djf_season.txt",
    sep="\t"
)

# ============================================================
# HAPUS NaN DAN DATA TIDAK VALID
# ============================================================
df = df.dropna(subset=["ssrd(W/m²)", "GHI"])

df = df[
    (df["ssrd(W/m²)"] > 0) &
    (df["GHI"] > 0)
]

# ============================================================
# DATA
# ============================================================
x = df["ssrd(W/m²)"]     # Measurement
y = df["GHI"]          # Model Perez

# ============================================================
# REGRESI LINEAR
# ============================================================
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# R² sesuai standar paper GHI
R2 = r_value**2

# ============================================================
# GARIS REGRESI
# ============================================================
x_line = np.linspace(0, 1100, 100)
y_reg = slope * x_line + intercept

# ============================================================
# PLOT
# ============================================================
plt.figure(figsize=(7,7))

# Scatter
plt.scatter(
    x,
    y,
    s=10,
    c="black",
    alpha=0.7,
    linewidths=0
)

# ============================================================
# GARIS 1:1
# ============================================================
plt.plot(
    [0,1100],
    [0,1100],
    color="red",
    linewidth=2,
    label="1:1"
)

# ============================================================
# GARIS REGRESI
# ============================================================
plt.plot(
    x_line,
    y_reg,
    color="blue",
    linewidth=2,
    label="Regression"
)

# ============================================================
# TAMPILKAN PERSAMAAN DAN R²
# ============================================================
plt.text(
    60,
    980,
    f"$R^2$ = {R2:.3f}\n"
    f"$y$ = {slope:.2f}$x$ + {intercept:.2f}",
    fontsize=13
)

# ============================================================
# JUDUL DAN LABEL
# ============================================================
plt.title(
    "Global Horizontal Irradiance (GHI) Semi-Empirical Perez (DJF Season)",
    fontsize=14
)

plt.xlabel(
    "Measurement (W/m²)",
    fontsize=14
)

plt.ylabel(
    "Model (W/m²)",
    fontsize=14
)

# ============================================================
# LIMIT
# ============================================================
plt.xlim(0,1200)
plt.ylim(0,1200)

# ============================================================
# GRID
# ============================================================
plt.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    color="black",
    alpha=0.5
)

# ============================================================
# STYLE
# ============================================================
plt.tick_params(
    direction="in",
    labelsize=12
)

ax = plt.gca()

for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.legend()

# ============================================================
# SAVE
# ============================================================
plt.savefig(
    "/data/mahasiswa/romadhon/perez1/scatterplot/global_horizontal_irradiance_djf_season.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# OUTPUT
# ============================================================
print("================================")
print("HASIL REGRESI")
print("================================")
print(f"Slope     = {slope:.4f}")
print(f"Intercept = {intercept:.4f}")
print(f"R         = {r_value:.4f}")
print(f"R²        = {R2:.4f}")
print(f"P-value   = {p_value:.4e}")
print("================================")