import xarray as xr
import pandas as pd
import numpy as np
import os


# ============================================================
# INPUT FILE NETCDF
# ============================================================

files = [
    #"/data/mahasiswa/romadhon/perez1/ghi/nc/ghi_december.nc",
    #"/data/mahasiswa/romadhon/perez1/ghi/nc/ghi_january.nc",
    #"/data/mahasiswa/romadhon/perez1/ghi/nc/ghi_february.nc",
    # "/data/mahasiswa/romadhon/perez1/ghi/nc/ghi_march.nc",
    # "/data/mahasiswa/romadhon/perez1/ghi/nc/ghi_april.nc",
    # "/data/mahasiswa/romadhon/perez1/ghi/nc/ghi_may.nc",
     "/data/mahasiswa/romadhon/perez1/ghi/nc/ghi_june.nc",
     "/data/mahasiswa/romadhon/perez1/ghi/nc/ghi_july.nc",
     "/data/mahasiswa/romadhon/perez1/ghi/nc/ghi_august.nc"#,
    # "/data/mahasiswa/romadhon/perez1/ghi/nc/ghi_september.nc",
    # "/data/mahasiswa/romadhon/perez1/ghi/nc/ghi_october.nc",
    # "/data/mahasiswa/romadhon/perez1/ghi/nc/ghi_november.nc"
]


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

output_dir = "/data/mahasiswa/romadhon/perez1/ghi/combined"
os.makedirs(output_dir, exist_ok=True)

output_nc = os.path.join(
    output_dir,
    "ghi_jja_season.nc"
)

output_txt = os.path.join(
    output_dir,
    "ghi_jja_season.txt"
)


# ============================================================
# BACA SEMUA FILE NETCDF
# ============================================================

df_list = []

for file in files:

    print(f"Membaca: {file}")

    ds = xr.open_dataset(file)

    df = ds.to_dataframe().reset_index()

    # Hapus index internal
    if "obs" in df.columns:
        df = df.drop(columns=["obs"])

    df_list.append(df)

    ds.close()


# ============================================================
# GABUNGKAN DATA
# ============================================================

df_merged = pd.concat(
    df_list,
    ignore_index=True
)


# ============================================================
# URUTKAN BERDASARKAN WAKTU
# ============================================================

if "valid_time" in df_merged.columns:

    df_merged["valid_time"] = pd.to_datetime(
        df_merged["valid_time"]
    )

    df_merged = df_merged.sort_values(
        by="valid_time"
    ).reset_index(drop=True)


# ============================================================
# URUTAN VARIABEL ASLI NETCDF
# ============================================================

column_order_nc = [
    "valid_time",
    "latitude",
    "longitude",
    "angstrom_turbidity",
    "reflectance",
    "sza_deg",
    "am",
    "epsilon",
    "I0_Wm2",
    "elevation_m",
    "tcwv_cm",
    "ssrd_Wm2",
    "Csat",
    "CI",
    "TL",
    "GHIC",
    "kc",
    "GHI"
]


# ============================================================
# CEK KOLOM NETCDF
# ============================================================

missing_columns = [
    col for col in column_order_nc
    if col not in df_merged.columns
]

if missing_columns:

    print("\nERROR: Kolom berikut tidak ditemukan:")

    for col in missing_columns:
        print(f"  - {col}")

    print("\nKolom yang tersedia:")
    print(df_merged.columns.tolist())

    raise ValueError(
        "Ada kolom yang tidak ditemukan."
    )


# ============================================================
# ATUR URUTAN DATA
# ============================================================

df_merged = df_merged[column_order_nc]


# ============================================================
# SIMPAN NETCDF
# ============================================================

df_nc = df_merged.copy()

df_nc.index.name = "obs"

ds_merged = xr.Dataset.from_dataframe(
    df_nc
)

ds_merged.to_netcdf(
    output_nc,
    format="NETCDF4"
)

print("\nNC berhasil disimpan:")
print(output_nc)


# ============================================================
# NAMA KOLOM KHUSUS UNTUK TXT
# ============================================================

txt_column_names = [
    "valid_time",
    "latitude",
    "longitude",
    "angstrom_turbidity(dimensionless)",
    "reflectance(dimensionless)",
    "sza(°)",
    "am(dimensionless)",
    "epsilon(dimensionless)",
    "I0(W/m²)",
    "elevation(m)",
    "tcwv(cm)",
    "ssrd(W/m²)",
    "Csat",
    "CI",
    "TL",
    "GHIC",
    "kc",
    "GHI"
]


# ============================================================
# BUAT DATAFRAME UNTUK TXT
# ============================================================

df_txt = df_merged.copy()

df_txt.columns = txt_column_names


# ============================================================
# SIMPAN TXT
# ============================================================

df_txt.to_csv(
    output_txt,
    sep="\t",
    index=False
)

print("\nTXT berhasil disimpan:")
print(output_txt)


# ============================================================
# INFORMASI HASIL
# ============================================================

print("\n==========================================")
print("PENGGABUNGAN MAM SELESAI")
print("==========================================")

print(f"Jumlah data  : {len(df_merged)}")
print(f"Jumlah kolom : {len(df_merged.columns)}")

print("\nUrutan variabel NC:")

for i, col in enumerate(df_merged.columns, 1):
    print(f"{i:2d}. {col}")

print("\nUrutan kolom TXT:")

for i, col in enumerate(df_txt.columns, 1):
    print(f"{i:2d}. {col}")

if "valid_time" in df_merged.columns:

    print(
        f"\nWaktu awal  : "
        f"{df_merged['valid_time'].min()}"
    )

    print(
        f"Waktu akhir : "
        f"{df_merged['valid_time'].max()}"
    )

print("\nOutput NC:")
print(output_nc)

print("\nOutput TXT:")
print(output_txt)


# ============================================================
# TUTUP DATASET
# ============================================================

ds_merged.close()