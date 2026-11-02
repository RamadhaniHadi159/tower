import pandas as pd
import numpy as np

# ============================================================
# LOAD DATA AWAL
# ============================================================
input_file = "/data/mahasiswa/romadhon/perez1/perez/perez_may.txt"

df = pd.read_csv(input_file, sep="\t")

# ============================================================
# 1. HITUNG CSAT
# ============================================================

C_raw = df["reflectance(dimensionless)"]
AM = df["am(dimensionless)"]
epsilon = df["epsilon(dimensionless)"]
theta_z = df["sza(°)"]

df["Csat"] = (
    (C_raw * AM * epsilon) / 2.283
) * ((90 - theta_z) ** (-0.26)) * np.exp(
    0.004 * (90 - theta_z)
)

df["Csat"] = df["Csat"].round(3)

# ============================================================
# 2. HITUNG CLOUD INDEX (CI)
# ============================================================

Csat_clean = df["Csat"].dropna()

sorted_csat = np.sort(Csat_clean)

print("\n===== STATISTIK CSAT =====")

print("Cmin lama :", np.mean(sorted_csat[1:5]))
print("Cmax lama :", np.mean(sorted_csat[-10:]))

print("P1  :", np.percentile(Csat_clean, 1))
print("P5  :", np.percentile(Csat_clean, 5))
print("P50 :", np.percentile(Csat_clean, 50))
print("P95 :", np.percentile(Csat_clean, 95))
print("P99 :", np.percentile(Csat_clean, 99))

print("Csat minimum :", np.min(Csat_clean))
print("Csat maksimum:", np.max(Csat_clean))

cmin = np.percentile(Csat_clean, 1)

cmax = np.percentile(Csat_clean, 99)

df["CI"] = (
    (df["Csat"] - cmin)
    / (cmax - cmin)
)

# batasi CI
df["CI"] = df["CI"].clip(
    lower=0,
    upper=1
)

print("\n===== STATISTIK CLOUD INDEX =====")
print(df["CI"].describe())

print(
    "\nJumlah CI=0 :",
    (df["CI"] == 0).sum()
)

print(
    "Jumlah CI=1 :",
    (df["CI"] == 1).sum()
)

df["CI"] = df["CI"].round(3)

# ============================================================
# 3. HITUNG LINKE TURBIDITY (TL)
# ============================================================

W = df["tcwv(cm)"] / 10
beta = df["angstrom_turbidity(dimensionless)"]

df["TL"] = (
    (
        1.8494
        + 0.2425 * W
        - 0.0203 * W**2
    )
    +
    (
        15.427
        + 0.3153 * W
        - 0.0254 * W**2
    )
    * beta
)

df["TL"] = df["TL"].round(3)

# ============================================================
# 4. HITUNG GHIC
# ============================================================

z = df["elevation(m)"]
theta_z = df["sza(°)"]
TL = df["TL"]
AM = df["am(dimensionless)"]
I_0 = df["I0(W/m²)"]

cos_theta_z = np.cos(np.radians(theta_z))

cg1 = 0.0000509 * z + 0.868
cg2 = 0.0000392 * z + 0.0387

fh1 = np.exp(-z / 8000)
fh2 = np.exp(-z / 1250)

df["GHIC"] = (
    cg1
    * I_0
    * cos_theta_z
    * np.exp(
        -cg2
        * AM
        * (fh1 + fh2 * (TL - 1))
    )
    * np.exp(0.01 * AM**1.8)
)

df["GHIC"] = df["GHIC"].round(3)

# ============================================================
# 5. HITUNG kc
# ============================================================

df["kc"] = (
    2.36 * df["CI"]**5
    - 6.2 * df["CI"]**4
    + 6.22 * df["CI"]**3
    - 2.63 * df["CI"]**2
    - 0.58 * df["CI"]
    + 1
)

df["kc"] = df["kc"].round(3)

# ============================================================
# 6. HITUNG GHI FINAL
# ============================================================

df["GHI"] = (
    df["kc"]
    * df["GHIC"]
    * (
        0.0001 * df["kc"] * df["GHIC"]
        + 0.9
    )
)

df["GHI"] = df["GHI"].round(3)

# ============================================================
# SIMPAN OUTPUT AKHIR
# ============================================================

output_file = "/data/mahasiswa/romadhon/perez1/ghi/ghi_may.txt"

df.to_csv(
    output_file,
    sep="\t",
    index=False
)

# ============================================================
# RINGKASAN HASIL
# ============================================================

print("======================================")
print("Cmin :", round(cmin, 5))
print("Cmax :", round(cmax, 5))
print("Output :", output_file)
print("======================================")

print(
    df[
        [
            "Csat",
            "CI",
            "TL",
            "GHIC",
            "kc",
            "GHI"
        ]
    ].head(20)
)