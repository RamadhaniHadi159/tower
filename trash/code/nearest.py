import os
import xarray as xr
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

GHI_FILE = (
    "/data/mahasiswa/romadhon/perez1/"
    "ghi/combined/nc/ghi_jja_season.nc"
)

OUTPUT_TXT = (
    "/data/mahasiswa/romadhon/perez1/"
    "ghi/bmkgjuanda/nearest_jja_bmkg.txt"
)

# Koordinat BMKG Stasiun Meteorologi Kelas I Juanda
BMKG_LAT = -7.3798
BMKG_LON = 112.7872


# ============================================================
# 1. BUAT OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_TXT),
    exist_ok=True
)


# ============================================================
# 2. BACA DATA GHI
# ============================================================

ds = xr.open_dataset(GHI_FILE)

df = (
    ds[
        [
            "valid_time",
            "latitude",
            "longitude",
            "ssrd_Wm2",
            "GHI"
        ]
    ]
    .to_dataframe()
    .reset_index()
)

ds.close()

print("Data berhasil dibaca")
print(f"Dataset obs: {len(df)}")


# ============================================================
# 3. PASTIKAN NAMA KOLOM GHI
# ============================================================

df = df.rename(
    columns={
        "ssrd_Wm2": "ssrd(W/m²)"
    }
)


# ============================================================
# 4. PASTIKAN VALID_TIME BERTIPE DATETIME
# ============================================================

df["valid_time"] = pd.to_datetime(
    df["valid_time"],
    errors="coerce"
)


# ============================================================
# 5. CARI GRID MASINGAR TERDEKAT
# ============================================================

grid = (
    df[
        [
            "latitude",
            "longitude"
        ]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

grid["distance"] = np.sqrt(
    (grid["latitude"] - BMKG_LAT) ** 2
    +
    (grid["longitude"] - BMKG_LON) ** 2
)

nearest_grid = (
    grid
    .sort_values("distance")
    .iloc[0]
)

nearest_lat = nearest_grid["latitude"]
nearest_lon = nearest_grid["longitude"]
nearest_distance = nearest_grid["distance"]


print("\n========================================")
print("NEAREST-NEIGHBOR")
print("========================================")

print(
    f"BMKG Juanda: "
    f"{BMKG_LAT:.6f}, {BMKG_LON:.6f}"
)

print(
    f"Grid Masingar terdekat: "
    f"{nearest_lat:.6f}, {nearest_lon:.6f}"
)

print(
    f"Jarak grid = "
    f"{nearest_distance:.6f}°"
)


# ============================================================
# 6. AMBIL DATA PADA GRID TERDEKAT
# ============================================================

df_nearest = df[
    (df["latitude"] == nearest_lat)
    &
    (df["longitude"] == nearest_lon)
].copy()


# ============================================================
# 7. URUTKAN BERDASARKAN WAKTU
# ============================================================

df_nearest = (
    df_nearest
    .sort_values("valid_time")
    .reset_index(drop=True)
)


# ============================================================
# 8. HAPUS DUPLIKAT WAKTU
# ============================================================

df_nearest = (
    df_nearest
    .drop_duplicates(
        subset=["valid_time"],
        keep="first"
    )
    .reset_index(drop=True)
)


print(
    f"Jumlah data pada grid tersebut = "
    f"{len(df_nearest)}"
)


# ============================================================
# 9. HAPUS SEMUA BARIS YANG MEMILIKI NaN
# ============================================================

print("\n========================================")
print("FILTER NaN")
print("========================================")

print(f"Jumlah data sebelum filter NaN : {len(df_nearest)}")

# Pastikan kolom numerik benar-benar bertipe numeric
df_nearest["ssrd(W/m²)"] = pd.to_numeric(
    df_nearest["ssrd(W/m²)"],
    errors="coerce"
)

df_nearest["GHI"] = pd.to_numeric(
    df_nearest["GHI"],
    errors="coerce"
)

# Hapus baris jika salah satu nilai adalah NaN
df_nearest = df_nearest[
    df_nearest["ssrd(W/m²)"].notna()
    &
    df_nearest["GHI"].notna()
].copy()

df_nearest = df_nearest.reset_index(drop=True)

print(f"Jumlah data setelah filter NaN : {len(df_nearest)}")


# ============================================================
# 10. CEK ULANG — TIDAK BOLEH ADA NaN
# ============================================================

print("\n========================================")
print("CEK NaN SETELAH FILTER")
print("========================================")

print(df_nearest.isna().sum())


# ============================================================
# 11. SUSUN KOLOM OUTPUT
# ============================================================

df_nearest = df_nearest[
    [
        "valid_time",
        "latitude",
        "longitude",
        "ssrd(W/m²)",
        "GHI"
    ]
].copy()


# ============================================================
# 12. CEK DATA SEBELUM DISIMPAN
# ============================================================

print("\n========================================")
print("DATA SEBELUM DISIMPAN")
print("========================================")

print(
    df_nearest.head(10).to_string(index=False)
)

print("\nJumlah data akhir:")
print(len(df_nearest))

print("\nJumlah NaN:")
print(df_nearest.isna().sum())


# ============================================================
# 13. SIMPAN OUTPUT — SUDAH BERSIH DARI NaN
# ============================================================

# Hapus file lama terlebih dahulu jika ada
if os.path.exists(OUTPUT_TXT):
    os.remove(OUTPUT_TXT)

df_nearest.to_csv(
    OUTPUT_TXT,
    sep="\t",
    index=False,
    float_format="%.3f"
)


# ============================================================
# 14. VERIFIKASI FILE HASIL
# ============================================================

print("\n========================================")
print("SELESAI")
print("========================================")

print(f"Output: {OUTPUT_TXT}")