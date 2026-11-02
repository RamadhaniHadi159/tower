import pandas as pd
import numpy as np

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("/data/dosen/ramsa/ramadhani/perez/TL1mounth.txt", sep="\t")

# Format time menjadi 2 digit
df["time"] = df["time"].astype(str).str.zfill(2)

# =========================
# INPUT DATA
# =========================

# elevasi (meter)
z = df["elevation"]

# solar zenith angle (derajat)
theta_z = df["sza"]

# Linke turbidity
TL = df["TL"]

# air mass
AM = df["am"]

# Solar constant
I_0 = df["I0"]

# =========================
# HITUNG COS THETA
# =========================
cos_theta_z = np.cos(np.radians(theta_z))

# =========================
# HITUNG PARAMETER PEREZ
# =========================

cg1 = 0.0000509 * z + 0.868

cg2 = 0.0000392 * z + 0.0387

fh1 = np.exp(-z / 8000)

fh2 = np.exp(-z / 1250)

# =========================
# HITUNG GHI
# =========================

GHIC = (
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

# Bulatkan 3 angka belakang koma
GHIC = GHIC.round(3)

# =========================
# SIMPAN HASIL
# =========================
df["GHIC"] = GHIC
df.to_csv("/data/dosen/ramsa/ramadhani/perez/GHIC1mounth.txt", sep="\t", index=False)

print(GHIC)