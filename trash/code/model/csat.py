import pandas as pd
import numpy as np

# Load data yang akan diolah
df = pd.read_csv("/data/dosen/ramsa/ramadhani/perez/perez1mounth.txt", sep="\t")

# Format time menjadi 2 digit
df["time"] = df["time"].astype(str).str.zfill(2)

# C_raw, nilai reflectance mentah 
# AM, masa udara
# ε, faktor jarak matahari - bumi,
# sza, sudut zenith matahari

C_raw = df["reflectance"] 
AM = df["am"]
epsilon = df["epsilon"]
theta_z = df["sza"]

# (90 - sza) > 0 # jika perlu difilter lagi

Csat = (
    (C_raw * AM * epsilon) / 2.283
) * ((90 - theta_z) ** (-0.26)) * np.exp(
    0.004 * (90 - theta_z)
)

# Bulatkan 3 angka belakang koma
Csat = Csat.round(3)

# Simpan hasil perhitungan Csat
df["Csat"] = Csat
df.to_csv("/data/dosen/ramsa/ramadhani/perez/Csat1mounth.txt", sep="\t", index=False) 

print(Csat)