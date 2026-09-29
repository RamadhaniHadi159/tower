import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("/data/dosen/ramsa/ramadhani/perez/GHIC1mounth.txt", sep="\t")

# Format time menjadi 2 digit
df["time"] = df["time"].astype(str).str.zfill(2)

# Hitung kcP dari CI
kc = (
    2.36 * df["CI"]**5
    - 6.2 * df["CI"]**4
    + 6.22 * df["CI"]**3
    - 2.63 * df["CI"]**2
    - 0.58 * df["CI"]
    + 1
)

# Bulatkan 3 angka belakang koma
kc = kc.round(3)

# Simpan hasil
df["kc"] = kc
df.to_csv("/data/dosen/ramsa/ramadhani/perez/kc1mounth.txt", sep="\t", index=False)

# Print hasil
print(df[["kc"]])