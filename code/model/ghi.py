import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("/data/dosen/ramsa/ramadhani/perez/kc1mounth.txt", sep="\t")

# Format time menjadi 2 digit
df["time"] = df["time"].astype(str).str.zfill(2)

# Hitung GHI final Perez
GHI = (
    df["kc"]
    * df["GHIC"]
    * (0.0001 * df["kc"] * df["GHIC"] + 0.9)
)

# Bulatkan 3 angka belakang koma
GHI = GHI.round(3)

# Simpan hasil
df["GHI"] = GHI
df.to_csv("/data/dosen/ramsa/ramadhani/perez/GHI1mounth.txt", sep="\t", index=False)

# Print hasil
print(df[["GHI"]])