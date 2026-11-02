import xarray as xr
import pandas as pd
import numpy as np
import os

from scipy.stats import linregress
from concurrent.futures import ProcessPoolExecutor

# ============================================================
# CPU PARALLEL
# ============================================================
n_cpu = int(
    os.environ.get(
        "SLURM_CPUS_PER_TASK",
        40
    )
)

print(f"CPU digunakan: {n_cpu}")

# ============================================================
# KONFIGURASI
# ============================================================
config = {
    "grid": True,
    "unit_conv": True,
    "drop_parameter": True,
    "remove_noise": False
}

# ============================================================
# 1. LOAD DATASET
# ============================================================
hima = xr.open_mfdataset(
    "/data/mahasiswa/romadhon/perez/himawari/himawari_trial/himawari_*.nc",
    engine="h5netcdf",
    combine="by_coords"
)

print("\n" + "=" * 70)
print("HIMAWARI-8")
print("=" * 70)

print(hima)

print("\n--- DIMENSIONS ---")
print(hima.dims)

print("\n--- COORDINATES ---")
print(hima.coords)

print("\n--- TIME ---")
print(hima.valid_time.values[:100])

print("\n--- LATITUDE ---")
print(hima.latitude.values[:10])

print("\n--- LONGITUDE ---")
print(hima.longitude.values[:10])

print("\n--- VARIABLE ---")
print(list(hima.data_vars))


era5ssrd = xr.open_dataset(
    "/data/mahasiswa/romadhon/perez/era5/ssrd/2022/dataera5ssrdjanuary.nc",
    engine="h5netcdf"
)

print("\n" + "=" * 70)
print("ERA5 SSRD")
print("=" * 70)

print(era5ssrd)

print("\n--- DIMENSIONS ---")
print(era5ssrd.dims)

print("\n--- COORDINATES ---")
print(era5ssrd.coords)

print("\n--- TIME ---")
print(era5ssrd.valid_time.values[:10])

print("\n--- LATITUDE ---")
print(era5ssrd.latitude.values[:10])

print("\n--- LONGITUDE ---")
print(era5ssrd.longitude.values[:10])

print("\n--- VARIABLE ---")
print(list(era5ssrd.data_vars))

era5tcwv = xr.open_dataset(
    "/data/mahasiswa/romadhon/perez/era5/tcwv/2022/dataera5tcwvjanuary.nc",
    engine="h5netcdf"
)


print("\n" + "=" * 70)
print("ERA5 TCWV")
print("=" * 70)

print(era5tcwv)

print("\n--- DIMENSIONS ---")
print(era5tcwv.dims)

print("\n--- COORDINATES ---")
print(era5tcwv.coords)

print("\n--- TIME ---")
print(era5tcwv.valid_time.values[:10])

print("\n--- LATITUDE ---")
print(era5tcwv.latitude.values[:10])

print("\n--- LONGITUDE ---")
print(era5tcwv.longitude.values[:10])

print("\n--- VARIABLE ---")
print(list(era5tcwv.data_vars))

# Gabungkan dataset ERA5
era5 = xr.merge([
    era5ssrd,
    era5tcwv
])

masingar = xr.open_dataset(
    "/data/mahasiswa/romadhon/perez/masingar/aerosol/2022/dataaerosoljanuary.nc",
    engine="h5netcdf"
)

print("\n" + "=" * 70)
print("MASINGAR")
print("=" * 70)

print(masingar)

print("\n--- DIMENSIONS ---")
print(masingar.dims)

print("\n--- COORDINATES ---")
print(masingar.coords)

print("\n--- VARIABLES ---")
print(list(masingar.data_vars))

print("\n===== INFORMASI MASINGAR =====")
print(masingar)

print("\n===== COORDINATES MASINGAR =====")
print(masingar.coords)

print("\n===== LATITUDE =====")
print(masingar.latitude.values[:10])

print("\n===== LONGITUDE =====")
print(masingar.longitude.values[:10])

print("\n===== LATITUDE ATTRS =====")
print(masingar.latitude.attrs)

print("\n===== LONGITUDE ATTRS =====")
print(masingar.longitude.attrs)

# ============================================================
# 2. SAMAKAN NAMA DIMENSI WAKTU
# ============================================================
if "valid_time" in era5.dims:
    era5 = era5.rename({"valid_time": "time"})

if "valid_time" in hima.dims:
    hima = hima.rename({"valid_time": "time"})

if "valid_time" in masingar.dims:
    masingar = masingar.rename({"valid_time": "time"})

# ============================================================
# 3. SORT LATITUDE
# ============================================================
hima = hima.sortby("latitude")
era5 = era5.sortby("latitude")
masingar = masingar.sortby("latitude")

# ============================================================
# 4. INTERPOLASI KE GRID CAMS
# ============================================================
if config["grid"]:

    print("Interpolasi Himawari → Masingar ...")

    hima_interp = hima.interp(
        time=masingar.time,
        latitude=masingar.latitude,
        longitude=masingar.longitude,
        method="nearest"
    )

    print("Interpolasi ERA5 → Masingar ...")

    era5_interp = era5.interp(
        time=masingar.time,
        latitude=masingar.latitude,
        longitude=masingar.longitude,
        method="nearest"
    )

else:
    hima_interp = hima
    era5_interp = era5

# ============================================================
# 5. GABUNGKAN SEMUA DATASET
# ============================================================
combined = xr.merge([
    masingar,
    hima_interp,
    era5_interp
])

# ============================================================
# 6. KONVERSI KE DATAFRAME
# ============================================================
df = (
    combined
    .to_dataframe()
    .reset_index()
)

# ============================================================
# 7. HAPUS KOLOM TIDAK PERLU
# ============================================================
drop_cols = [
    "number",
    "expver"
]

df = df.drop(
    columns=[
        col for col in drop_cols
        if col in df.columns
    ],
    errors="ignore"
)

# ============================================================
# 8. FORMAT WAKTU
# ============================================================
df["time"] = pd.to_datetime(df["time"])

# DATE -> 20221201
df["date"] = (
    df["time"]
    .dt.strftime("%Y%m%d")
    .astype(int)
)

# TIME -> 00, 03, 06, ...
df["time"] = (
    df["time"]
    .dt.strftime("%H")
)

# ============================================================
# 9. KONVERSI SSRD
# ============================================================
if config["unit_conv"]:

    df["ssrd_wm2"] = (
        df["ssrd"] / 3600
    ).round(3)

# ============================================================
# 9A. HAPUS DATA TIDAK VALID
# ============================================================

valid_cols = [
    "radiance",
    "reflectance",
    "ssrd",
    "ssrd_wm2"
]

# ubah inf menjadi NaN
df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

# hapus NaN pada kolom penting
df = df.dropna(
    subset=valid_cols
)

# hapus nilai nol atau negatif
df = df[
    (df["radiance"] > 0) &
    (df["reflectance"] > 0) &
    (df["ssrd"] > 0) &
    (df["ssrd_wm2"] > 0)
].copy()

# ============================================================
# FILTER SOLAR ZENITH
# ============================================================
df = df[
    df["sza"] < 80
].copy()

print("===================================")
print("Filter SZA < 80 selesai")
print(f"Jumlah data tersisa : {len(df):,}")
print("===================================")

# ============================================================
# CEK DATA SETELAH FILTER
# ============================================================
print("\n===== CEK DATA =====")

print("Shape :", df.shape)

print(
    df[
        ["date", "time", "latitude", "longitude"]
    ]
    .drop_duplicates()
    .shape
)

print(
    df.groupby(
        ["date", "time"]
    )
    .size()
    .head(20)
)
    
# ============================================================
# 10. PANJANG GELOMBANG
# ============================================================
wavelengths = np.array([
    0.469,
    0.550,
    0.670,
    0.865,
    1.240
])

ln_lambda = np.log(wavelengths)

# ============================================================
# 11. HITUNG ANGSTROM
# ============================================================
def hitung_angstrom(row):

    try:

        aod = np.array([
            row["aod469"],
            row["aod550"],
            row["aod670"],
            row["aod865"],
            row["aod1240"]
        ])

        if (
            np.any(aod <= 0)
            or np.any(np.isnan(aod))
            or np.any(np.isinf(aod))
        ):
            return np.nan, np.nan

        ln_tau = np.log(aod)

        slope, intercept, *_ = linregress(
            ln_lambda,
            ln_tau
        )

        alpha = -slope
        beta = np.exp(intercept)

        return (
            round(alpha, 3),
            round(beta, 3)
        )

    except:
        return np.nan, np.nan

# ============================================================
# 12. PARALLEL PROCESS
# ============================================================
with ProcessPoolExecutor(
    max_workers=n_cpu
) as executor:

    results = list(
        executor.map(
            hitung_angstrom,
            [
                row
                for _, row in df.iterrows()
            ]
        )
    )

# ============================================================
# 13. MASUKKAN α & β
# ============================================================
df["alpha"] = [
    r[0]
    for r in results
]

df["beta"] = [
    r[1]
    for r in results
]

# ============================================================
# 14. BULATKAN NUMERIK
# ============================================================
numeric_cols = df.select_dtypes(
    include=[np.number]
).columns

df[numeric_cols] = (
    df[numeric_cols]
    .round(3)
)

# ============================================================
# 15. URUTKAN KOLOM
# ============================================================
main_cols = [
    "date",
    "time",
    "latitude",
    "longitude"
]

other_cols = [
    col
    for col in df.columns
    if col not in main_cols
]

df_out = df[
    main_cols + other_cols
]

# ============================================================
# 16. OPTIONAL FILTER
# ============================================================
if config["drop_parameter"]:

    df_out = df_out.drop(
        columns=[
            "aod469",
            "aod550",
            "aod670",
            "aod865",
            "aod1240",
            "radiance",
            "ssrd"
        ],
        errors="ignore"
    )

if config["remove_noise"]:

    df_out = df_out[
        df_out["ssrd_wm2"] > 0
    ]

# ============================================================
# 17. SORT DATA
# ============================================================
df_out = df_out.sort_values(
    by=[
        "date",
        "time",
        "latitude",
        "longitude"
    ],
    ascending=[
        True,
        True,
        False,
        True
    ]
)

# ============================================================
# 18. SIMPAN OUTPUT
# ============================================================
output_csv = (
    "/data/mahasiswa/"
    "romadhon/perez/"
    "perezdecembertes.txt"
)

df_out.to_csv(
    output_csv,
    sep="\t",
    index=False
)

print("===================================")
print("✔ Berhasil!")
print("Output :", output_csv)
print("===================================")