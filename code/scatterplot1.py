import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ============================================================
# 1. BACA DATA HASIL QC
# ============================================================

file_input = (
    "/data/mahasiswa/romadhon/perez1/"
    "ghi/bmkgjuanda/nearest_jja_bmkg.txt"
)

df = pd.read_csv(
    file_input,
    sep="\t"
)

print("================================")
print("DATA AWAL")
print("================================")
print(f"Jumlah data awal : {len(df)}")

print("\nKolom:")
print(df.columns.tolist())


# ============================================================
# 2. HANYA BUANG DATA YANG NaN
#    QC utama sudah dilakukan pada pipeline Perez
# ============================================================

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(
    subset=[
        "ssrd(W/m²)",
        "GHI"
    ]
).copy()


# ============================================================
# 3. DATA SETELAH QC
# ============================================================

print("\n================================")
print("DATA SETELAH QC")
print("================================")
print(f"Jumlah data : {len(df)}")

print("\nStatistik ERA5 SSRD:")
print(df["ssrd(W/m²)"].describe())

print("\nStatistik Perez GHI:")
print(df["GHI"].describe())


# ============================================================
# 4. VARIABEL REGRESI
# ============================================================

# X = ERA5 SSRD
# Y = GHI hasil model Perez

x = df["ssrd(W/m²)"].to_numpy()
y = df["GHI"].to_numpy()


# ============================================================
# 5. REGRESI LINEAR
# ============================================================

slope, intercept, r_value, p_value, std_err = linregress(x, y)

R2 = r_value ** 2

# RMSE
RMSE = np.sqrt(np.mean((y - x) ** 2))

# MBE
MBE = np.mean(y - x)

# Relative MBE
mean_obs = np.mean(x)

if mean_obs != 0:
    rMBE = (MBE / mean_obs) * 100
else:
    rMBE = np.nan

# Relative RMSE
if mean_obs != 0:
    rRMSE = (RMSE / mean_obs) * 100
else:
    rRMSE = np.nan


# ============================================================
# 6. GARIS REGRESI
# ============================================================

x_min = 0
x_max = 1200

x_line = np.linspace(x_min, x_max, 200)

y_reg = slope * x_line + intercept


# ============================================================
# 7. SCATTERPLOT
# ============================================================

plt.figure(figsize=(7, 7))

plt.scatter(
    x,
    y,
    s=10,
    c="black",
    alpha=0.7,
    linewidths=0
)

# ------------------------------------------------------------
# Garis 1:1
# ------------------------------------------------------------

plt.plot(
    [x_min, x_max],
    [x_min, x_max],
    color="red",
    linewidth=2,
    label="1:1"
)

# ------------------------------------------------------------
# Garis regresi
# ------------------------------------------------------------

plt.plot(
    x_line,
    y_reg,
    color="blue",
    linewidth=2,
    label="Regression"
)


# ============================================================
# 8. INFORMASI REGRESI PADA GRAFIK
# ============================================================

text = (
    f"$R^2$ = {R2:.3f}\n"
    f"$y$ = {slope:.2f}$x$ + {intercept:.2f}"
)

plt.text(
    60,
    980,
    text,
    fontsize=13
)


# ============================================================
# 9. LABEL DAN FORMAT GRAFIK
# ============================================================

plt.title(
    "Global Horizontal Irradiance (GHI) "
    "Semi-Empirical Perez (JJA BMKG Juanda)",
    fontsize=14
)

plt.xlabel(
    "ERA5 SSRD (W/m²)",
    fontsize=14
)

plt.ylabel(
    "Perez GHI (W/m²)",
    fontsize=14
)

plt.xlim(x_min, x_max)
plt.ylim(0, 1200)

plt.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    color="black",
    alpha=0.5
)

plt.tick_params(
    direction="in",
    labelsize=12
)

ax = plt.gca()

for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.legend()

plt.tight_layout()


# ============================================================
# 10. SIMPAN GAMBAR
# ============================================================

output_file = (
    "/data/mahasiswa/romadhon/perez1/"
    "grafikscatterplot/global_horizontal_irradiance_jja_bmkg_juanda.pdf"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 11. HASIL REGRESI
# ============================================================

print("\n================================")
print("HASIL REGRESI")
print("================================")

print(f"Jumlah data = {len(df)}")

print(f"Slope       = {slope:.4f}")
print(f"Intercept   = {intercept:.4f}")
print(f"R           = {r_value:.4f}")
print(f"R²          = {R2:.4f}")
print(f"P-value     = {p_value:.4e}")

print("--------------------------------")
print("ERROR METRICS")
print("--------------------------------")

print(f"MBE         = {MBE:.4f} W/m²")
print(f"rMBE        = {rMBE:.4f} %")
print(f"RMSE        = {RMSE:.4f} W/m²")
print(f"rRMSE       = {rRMSE:.4f} %")

print("================================")
print(f"Grafik disimpan:")
print(output_file)
print("================================")