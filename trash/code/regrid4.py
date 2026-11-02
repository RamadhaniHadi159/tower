import xarray as xr
import pandas as pd
import numpy as np
import os
import xesmf as xe


# ============================================================
# KONFIGURASI
# ============================================================

CONFIG = {
    "spatial_matching": True,
    
    "temporal_matching": True,

    "unit_conv": True,

    "min_himawari_scene": 5,
    
    "min_ssrd_wm2": 100.0,
    
    "max_sza": 80.0,

    "output_file":
        "/data/mahasiswa/romadhon/perez1/"
        "perez_december_2021.txt"
}


# ============================================================
# CPU
# ============================================================

n_cpu = int(
    os.environ.get(
        "SLURM_CPUS_PER_TASK",
        40
    )
)

print("=" * 70)
print(f"CPU tersedia : {n_cpu}")
print("=" * 70)


# ============================================================
# 1. LOAD HIMAWARI
# ============================================================

hima = xr.open_mfdataset(
    "/data/mahasiswa/romadhon/perez1/"
    "himawari/2021/himawari_december/.nc/"
    "himawari_*.nc",

    engine="h5netcdf",
    combine="by_coords"
)

print("\n" + "=" * 70)
print("HIMAWARI-8")
print("=" * 70)

print(hima)

print("\nVARIABLE HIMAWARI:")
print(list(hima.data_vars))


# ============================================================
# 2. LOAD ERA5 SSRD
# ============================================================

era5ssrd = xr.open_dataset(
    "/data/mahasiswa/romadhon/perez1/"
    "era5/ssrd/2021/"
    "dataera5ssrddecember.nc",

    engine="h5netcdf"
)

print("\n" + "=" * 70)
print("ERA5 SSRD")
print("=" * 70)

print(era5ssrd)

print("\nVARIABLE ERA5 SSRD:")
print(list(era5ssrd.data_vars))


# ============================================================
# 3. LOAD ERA5 TCWV
# ============================================================

era5tcwv = xr.open_dataset(
    "/data/mahasiswa/romadhon/perez1/"
    "era5/tcwv/2021/"
    "dataera5tcwvdecember.nc",

    engine="h5netcdf"
)

print("\n" + "=" * 70)
print("ERA5 TCWV")
print("=" * 70)

print(era5tcwv)

print("\nVARIABLE ERA5 TCWV:")
print(list(era5tcwv.data_vars))


# ============================================================
# 4. PERBAIKI LONGITUDE ERA5 TCWV
# ============================================================

print("\n" + "=" * 70)
print("CHECK LONGITUDE ERA5")
print("=" * 70)

print(
    "SSRD longitude :",
    float(era5ssrd.longitude.min()),
    "→",
    float(era5ssrd.longitude.max())
)

print(
    "TCWV longitude :",
    float(era5tcwv.longitude.min()),
    "→",
    float(era5tcwv.longitude.max())
)


# ------------------------------------------------------------
# Normalisasi longitude
# ------------------------------------------------------------

era5tcwv = era5tcwv.assign_coords(
    longitude=(
        ((era5tcwv.longitude + 180) % 360) - 180
    )
)


# ------------------------------------------------------------
# Crop wilayah penelitian
# ------------------------------------------------------------

era5tcwv = era5tcwv.sel(
    longitude=slice(110, 116.5)
)


# ------------------------------------------------------------
# Sort longitude
# ------------------------------------------------------------

era5tcwv = era5tcwv.sortby(
    "longitude"
)


print(
    "TCWV longitude setelah diperbaiki :",
    float(era5tcwv.longitude.min()),
    "→",
    float(era5tcwv.longitude.max())
)


# ============================================================
# 5. GABUNG ERA5
# ============================================================

era5 = xr.merge(
    [
        era5ssrd,
        era5tcwv
    ],
    compat="override"
)

print("\n" + "=" * 70)
print("ERA5 BERHASIL DIGABUNGKAN")
print("=" * 70)

print(era5)


# ============================================================
# 6. LOAD MASINGAR
#
# Alpha dan beta SUDAH tersedia.
# Tidak dihitung ulang.
# ============================================================

masingar = xr.open_dataset(
    "/data/mahasiswa/romadhon/perez1/"
    "masingar/aerosol/2021/"
    "dataaerosoldecember.nc",

    engine="h5netcdf"
)

print("\n" + "=" * 70)
print("MASINGAR")
print("=" * 70)

print(masingar)

print("\nVARIABLE MASINGAR:")
print(list(masingar.data_vars))


# ============================================================
# 7. SAMAKAN NAMA DIMENSI WAKTU
# ============================================================

def rename_time(ds):

    if "valid_time" in ds.dims:

        ds = ds.rename({
            "valid_time": "time"
        })

    elif "valid_time" in ds.coords:

        ds = ds.rename({
            "valid_time": "time"
        })

    return ds


hima = rename_time(hima)
era5 = rename_time(era5)
masingar = rename_time(masingar)


# ============================================================
# 8. PASTIKAN TIME DATETIME
# ============================================================

hima["time"] = pd.to_datetime(
    hima["time"].values
)

era5["time"] = pd.to_datetime(
    era5["time"].values
)

masingar["time"] = pd.to_datetime(
    masingar["time"].values
)


# ============================================================
# 9. SORT
# ============================================================

hima = hima.sortby("time")

era5 = era5.sortby("time")

masingar = masingar.sortby("time")


# ============================================================
# 10. CEK DUPLIKAT TIME
# ============================================================

print("\n" + "=" * 70)
print("CHECK TIME")
print("=" * 70)

hima_time_index = pd.DatetimeIndex(
    hima.time.values
)

era5_time_index = pd.DatetimeIndex(
    era5.time.values
)

masingar_time_index = pd.DatetimeIndex(
    masingar.time.values
)


print(
    "Himawari timestamp:",
    len(hima_time_index)
)

print(
    "Himawari unique timestamp:",
    hima_time_index.nunique()
)

print(
    "ERA5 timestamp:",
    len(era5_time_index)
)

print(
    "ERA5 unique timestamp:",
    era5_time_index.nunique()
)

print(
    "Masingar timestamp:",
    len(masingar_time_index)
)

print(
    "Masingar unique timestamp:",
    masingar_time_index.nunique()
)


# ============================================================
# 11. CELL BOUNDS
# ============================================================

def create_bounds(values):

    values = np.asarray(values)

    bounds = np.empty(
        len(values) + 1,
        dtype=float
    )

    bounds[1:-1] = (
        values[:-1] +
        values[1:]
    ) / 2

    bounds[0] = (
        values[0]
        -
        (values[1] - values[0]) / 2
    )

    bounds[-1] = (
        values[-1]
        +
        (values[-1] - values[-2]) / 2
    )

    return bounds


def add_bounds(ds):

    ds = ds.copy()

    lat_b = create_bounds(
        ds.latitude.values
    )

    lon_b = create_bounds(
        ds.longitude.values
    )

    ds = ds.assign_coords({

        "lat_b": (
            "lat_b",
            lat_b
        ),

        "lon_b": (
            "lon_b",
            lon_b
        )
    })

    return ds


# ============================================================
# 12. TARGET GRID = MASINGAR
# ============================================================

target_grid = xr.Dataset({

    "latitude": (
        ["latitude"],
        masingar.latitude.values
    ),

    "longitude": (
        ["longitude"],
        masingar.longitude.values
    )
})

target_grid = add_bounds(
    target_grid
)


print("\n" + "=" * 70)
print("TARGET GRID MASINGAR")
print("=" * 70)

print(
    "Jumlah latitude  :",
    len(target_grid.latitude)
)

print(
    "Jumlah longitude :",
    len(target_grid.longitude)
)


# ============================================================
# 13. SPATIAL CONSERVATIVE REGRIDDING
# ============================================================

if CONFIG["spatial_matching"]:

    print("\n" + "=" * 70)
    print("SPATIAL MATCHING")
    print("Metode : CONSERVATIVE")
    print("Target : GRID MASINGAR")
    print("=" * 70)


    # ========================================================
    # HIMAWARI → MASINGAR
    # ========================================================

    hima_grid = xr.Dataset({

        "latitude": (
            ["latitude"],
            hima.latitude.values
        ),

        "longitude": (
            ["longitude"],
            hima.longitude.values
        )
    })

    hima_grid = add_bounds(
        hima_grid
    )


    print(
        "\nRemapping Himawari → Masingar..."
    )

    regridder_hima = xe.Regridder(
        hima_grid,
        target_grid,
        method="conservative",
        periodic=False,
        reuse_weights=False
    )


    hima_spatial = regridder_hima(
        hima
    )


    print(
        "✓ Himawari selesai"
    )


    # ========================================================
    # ERA5 → MASINGAR
    # ========================================================

    era5_grid = xr.Dataset({

        "latitude": (
            ["latitude"],
            era5.latitude.values
        ),

        "longitude": (
            ["longitude"],
            era5.longitude.values
        )
    })

    era5_grid = add_bounds(
        era5_grid
    )


    print(
        "\nRemapping ERA5 → Masingar..."
    )

    regridder_era5 = xe.Regridder(
        era5_grid,
        target_grid,
        method="conservative",
        periodic=False,
        reuse_weights=False
    )


    era5_spatial = regridder_era5(
        era5
    )


    print(
        "✓ ERA5 selesai"
    )


else:

    hima_spatial = hima
    era5_spatial = era5


# ============================================================
# 14. COVERAGE TEMPORAL HIMAWARI
#
# 6/6 → DIGUNAKAN
# 5/6 → DIGUNAKAN
# <5/6 → DIBUANG
#
# TIDAK ADA INTERPOLASI
# ============================================================

print("\n" + "=" * 70)
print("COVERAGE TEMPORAL HIMAWARI")
print("=" * 70)


hima_time = pd.DatetimeIndex(
    hima.time.values
)


# ------------------------------------------------------------
# Awal jam
# ------------------------------------------------------------

hour_time = hima_time.floor(
    "1h"
)


# ------------------------------------------------------------
# Jumlah scene per jam
# ------------------------------------------------------------

scene_count = (
    pd.Series(
        1,
        index=hour_time
    )
    .groupby(level=0)
    .sum()
)


print(
    "\nDistribusi jumlah scene per jam:"
)

print(
    scene_count
    .value_counts()
    .sort_index()
)


print(
    "\nJumlah jam 6/6 :",
    int((scene_count == 6).sum())
)

print(
    "Jumlah jam 5/6 :",
    int((scene_count == 5).sum())
)

print(
    "Jumlah jam <5/6:",
    int((scene_count < 5).sum())
)


# ------------------------------------------------------------
# Jam valid
# ------------------------------------------------------------

valid_hours = scene_count[
    scene_count >=
    CONFIG["min_himawari_scene"]
].index


# ============================================================
# 15. TEMPORAL AVERAGING HIMAWARI
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL AVERAGING HIMAWARI")
print("=" * 70)


hima_hourly = (
    hima_spatial
    .resample(
        time="1h"
    )
    .mean()
)


# ------------------------------------------------------------
# Hanya 5/6 dan 6/6
# ------------------------------------------------------------

hima_hourly = hima_hourly.sel(
    time=hima_hourly.time.isin(
        valid_hours
    )
)


print(
    "Jumlah jam Himawari setelah "
    "coverage QC :",
    len(hima_hourly.time)
)


# ============================================================
# 16. MATCH HIMAWARI → MASINGAR
# ============================================================

hima_temporal = (
    hima_hourly
    .reindex(
        time=masingar.time
    )
)


# ============================================================
# 17. ERA5 TEMPORAL MATCHING
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL MATCHING ERA5")
print("=" * 70)


era5_temporal = (
    era5_spatial
    .reindex(
        time=masingar.time,
        method="nearest",
        tolerance=pd.Timedelta(
            "30min"
        )
    )
)


print(
    "✓ ERA5 temporal matching selesai"
)


# ============================================================
# 18. SAMAKAN KOORDINAT
# ============================================================

hima_temporal = hima_temporal.assign_coords({

    "latitude":
        masingar.latitude,

    "longitude":
        masingar.longitude
})


era5_temporal = era5_temporal.assign_coords({

    "latitude":
        masingar.latitude,

    "longitude":
        masingar.longitude
})


# ============================================================
# 19. MERGE
# ============================================================

print("\n" + "=" * 70)
print("MERGE DATA")
print("=" * 70)


combined = xr.merge(
    [
        masingar,
        hima_temporal,
        era5_temporal
    ],
    compat="override"
)


print(
    "✓ Semua dataset berhasil digabungkan"
)


# ============================================================
# 20. DATAFRAME
# ============================================================

df = (
    combined
    .to_dataframe()
    .reset_index()
)


print(
    "\nShape sebelum QC :",
    df.shape
)


# ============================================================
# 21. HAPUS KOLOM TIDAK DIPERLUKAN
# ============================================================

drop_cols = [
    "number",
    "expver"
]


df = df.drop(
    columns=[
        col
        for col in drop_cols
        if col in df.columns
    ],
    errors="ignore"
)


# ============================================================
# 22. VALID TIME
# ============================================================

df["valid_time"] = pd.to_datetime(
    df["time"],
    errors="coerce"
)


# ============================================================
# 23. KONVERSI SSRD
# ============================================================

if CONFIG["unit_conv"]:

    df["ssrd_wm2"] = (
        df["ssrd"] / 3600.0
    )


# ============================================================
# 24. QUALITY CONTROL
# ============================================================

print("\n" + "=" * 70)
print("QUALITY CONTROL")
print("=" * 70)


jumlah_awal = len(df)


# ============================================================
# QC 1 — VALID TIME
# ============================================================

sebelum = len(df)

df = df[
    df["valid_time"].notna()
].copy()


print(
    "QC valid_time :",
    sebelum - len(df),
    "data dibuang"
)


# ============================================================
# QC 2 — KOORDINAT VALID
#
# Wilayah Jawa Timur yang digunakan:
# latitude  -9 sampai -5
# longitude 110 sampai 116.5
# ============================================================

sebelum = len(df)

df = df[
    df["latitude"].between(-9.0, -5.0)
    &
    df["longitude"].between(110.0, 116.5)
].copy()


print(
    "QC koordinat wilayah :",
    sebelum - len(df),
    "data dibuang"
)


# ============================================================
# QC 3 — INF → NaN
# ============================================================

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)


# ============================================================
# QC 4 — RADIANCE
#
# Radiance harus finite dan > 0
# ============================================================

if "radiance" in df.columns:

    sebelum = len(df)

    df = df[
        np.isfinite(df["radiance"])
        &
        (df["radiance"] > 0)
    ].copy()

    print(
        "QC radiance finite & > 0 :",
        sebelum - len(df),
        "data dibuang"
    )


# ============================================================
# QC 5 — REFLECTANCE
#
# Reflectance harus finite dan > 0
# ============================================================

if "reflectance" in df.columns:

    sebelum = len(df)

    df = df[
        np.isfinite(df["reflectance"])
        &
        (df["reflectance"] > 0)
    ].copy()

    print(
        "QC reflectance finite & > 0 :",
        sebelum - len(df),
        "data dibuang"
    )


# ============================================================
# QC 6 — SZA
#
# SZA < 80°
# ============================================================

if "sza" in df.columns:

    sebelum = len(df)

    df = df[
        np.isfinite(df["sza"])
        &
        (df["sza"] < CONFIG["max_sza"])
    ].copy()

    print(
        "QC SZA < 80° :",
        sebelum - len(df),
        "data dibuang"
    )


# ============================================================
# QC 7 — SSRD
#
# SSRD >= 100 W/m²
#
# Nilai negatif / NaN otomatis terbuang.
# ============================================================

if "ssrd_wm2" in df.columns:

    sebelum = len(df)

    df = df[
        np.isfinite(df["ssrd_wm2"])
        &
        (
            df["ssrd_wm2"]
            >=
            CONFIG["min_ssrd_wm2"]
        )
    ].copy()

    print(
        "QC SSRD >= 100 W/m² :",
        sebelum - len(df),
        "data dibuang"
    )


# ============================================================
# QC 8 — TCWV
#
# Untuk sementara:
# hanya memastikan finite.
#
# Belum diberi batas maksimum karena
# threshold fisik belum kita sepakati.
# ============================================================

if "tcwv" in df.columns:

    sebelum = len(df)

    df = df[
        np.isfinite(df["tcwv"])
    ].copy()

    print(
        "QC TCWV finite :",
        sebelum - len(df),
        "data dibuang"
    )


# ============================================================
# QC 9 — AOD MASINGAR
#
# AOD harus > 0
# ============================================================

aod_cols = [
    "od500aer",
    "od550aer",
    "od870aer"
]


for col in aod_cols:

    if col in df.columns:

        sebelum = len(df)

        df = df[
            np.isfinite(df[col])
            &
            (df[col] > 0)
        ].copy()

        print(
            f"QC {col} finite & > 0 :",
            sebelum - len(df),
            "data dibuang"
        )


# ============================================================
# QC 10 — ANGSTROM EXPONENT
#
# Tidak dihitung ulang.
# Hanya memastikan nilai finite.
# ============================================================

if "angstrom_exponent" in df.columns:

    sebelum = len(df)

    df = df[
        np.isfinite(
            df["angstrom_exponent"]
        )
    ].copy()

    print(
        "QC Angstrom exponent finite :",
        sebelum - len(df),
        "data dibuang"
    )


# ============================================================
# QC 11 — ANGSTROM TURBIDITY
#
# Tidak dihitung ulang.
# Hanya memastikan nilai finite.
# ============================================================

if "angstrom_turbidity" in df.columns:

    sebelum = len(df)

    df = df[
        np.isfinite(
            df["angstrom_turbidity"]
        )
    ].copy()

    print(
        "QC Angstrom turbidity finite :",
        sebelum - len(df),
        "data dibuang"
    )


# ============================================================
# QC 12 — WATER VAPOUR / TCWV NON-NEGATIF
#
# Jika TCWV tersedia, secara fisik tidak boleh negatif.
# ============================================================

if "tcwv" in df.columns:

    sebelum = len(df)

    df = df[
        df["tcwv"] >= 0
    ].copy()

    print(
        "QC TCWV >= 0 :",
        sebelum - len(df),
        "data dibuang"
    )


# ============================================================
# 25. INFORMASI COVERAGE HIMAWARI
# ============================================================

coverage_df = (
    scene_count
    .rename(
        "himawari_scene_count"
    )
    .reset_index()
    .rename(
        columns={
            "index": "valid_time"
        }
    )
)


coverage_df[
    "himawari_coverage"
] = (
    coverage_df[
        "himawari_scene_count"
    ]
    .astype(int)
    .astype(str)
    + "/6"
)


df = df.merge(
    coverage_df,
    on="valid_time",
    how="left"
)


# ============================================================
# 26. FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL QC")
print("=" * 70)


print(
    "Jumlah data awal  :",
    f"{jumlah_awal:,}"
)

print(
    "Jumlah data akhir :",
    f"{len(df):,}"
)

print(
    "Data terbuang     :",
    f"{jumlah_awal - len(df):,}"
)


if jumlah_awal > 0:

    print(
        "Persentase lolos  :",
        f"{len(df) / jumlah_awal * 100:.2f}%"
    )


# ============================================================
# 27. DISTRIBUSI COVERAGE
# ============================================================

if "himawari_coverage" in df.columns:

    print(
        "\nDistribusi coverage Himawari "
        "pada data akhir:"
    )

    print(
        df[
            "himawari_coverage"
        ]
        .value_counts()
        .sort_index()
    )


# ============================================================
# 28. CEK VALID TIME
# ============================================================

print("\n" + "=" * 70)
print("CONTOH VALID TIME")
print("=" * 70)


show_cols = [
    "valid_time",
    "latitude",
    "longitude"
]


if "himawari_coverage" in df.columns:

    show_cols.append(
        "himawari_coverage"
    )


print(
    df[
        show_cols
    ]
    .drop_duplicates()
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# 29. SORT DATA
# ============================================================

df = df.sort_values(
    by=[
        "valid_time",
        "latitude",
        "longitude"
    ],
    ascending=[
        True,
        False,
        True
    ]
).reset_index(
    drop=True
)


# ============================================================
# 30. SIMPAN TXT
# ============================================================

output_file = CONFIG[
    "output_file"
]


df.to_csv(
    output_file,
    sep="\t",
    index=False
)
