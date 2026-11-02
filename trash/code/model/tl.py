import pandas as pd
import numpy as np

# Load data yang akan diolah
df = pd.read_csv("/data/dosen/ramsa/ramadhani/perez/CI1mounth.txt", sep="\t")

# Format time menjadi 2 digit
df["time"] = df["time"].astype(str).str.zfill(2)

W = df["tcwv"] / 10 # dirubah ke cm satuannya
beta = df["beta"]

TL = (
    1.8494
    + 0.2425 * W
    - 0.0203 * W**2
) + (
    15.427
    + 0.3153 * W
    - 0.0254 * W**2
) * beta

# Bulatkan 3 angka belakang koma
TL = TL.round(3)

# Simpan hasil perhitungan TL
df["TL"] = TL
df.to_csv("/data/dosen/ramsa/ramadhani/perez/TL1mounth.txt", sep="\t", index=False)

print(df["TL"])