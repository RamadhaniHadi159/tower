# ============================================================
# REGRID5.PY
#
# INTEGRASI:
#   HIMAWARI-8
#   ERA5 SSRD
#   ERA5 TCWV
#   MASINGAR AEROSOL
#
# PERIODE:
#   DESEMBER 2021
#
# SPATIAL:
#   Conservative / area-weighted menggunakan xESMF
#
# TEMPORAL:
#   Himawari 10 menit -> 1 jam
#   Coverage minimum 5 dari 6 scene per jam
#
# TARGET GRID:
#   Masingar
#   12 latitude x 17 longitude
#
# OUTPUT FINAL:
#   valid_time
#   latitude
#   longitude
#   angstrom_exponent
#   angstrom_turbidity
#   reflectance
#   sza
#   am
#   epsilon
#   I0
#   elevation
#   tcwv
#   ssrd
#
# CATATAN:
#   ssrd pada output sudah dikonversi menjadi W/m2
#   dari ERA5 accumulated solar radiation (J/m2)
#
# OUTPUT:
#   TXT SAJA
# ============================================================


import os
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe


# ============================================================
# 1. CONFIGURATION
# ============================================================

N_CPU = 40

CONFIG = {

    # --------------------------------------------------------
    # HIMAWARI DESEMBER 2021
    # --------------------------------------------------------
    "himawari": (
        "/data/mahasiswa/romadhon/perez1/"
        "himawari/2021/himawari_december/.nc/"
        "himawari_*.nc"
    ),

    # --------------------------------------------------------
    # ERA5 SSRD DESEMBER 2021
    # --------------------------------------------------------
    "era5_ssrd": (
        "/data/mahasiswa/romadhon/perez1/"
        "era5/ssrd/2021/"
        "dataera5ssrddecember.nc"
    ),

    # --------------------------------------------------------
    # ERA5 TCWV DESEMBER 2021
    # --------------------------------------------------------
    "era5_tcwv": (
        "/data/mahasiswa/romadhon/perez1/"
        "era5/tcwv/2021/"
        "dataera5tcwvdecember.nc"
    ),

    # --------------------------------------------------------
    # MASINGAR DESEMBER 2021
    # --------------------------------------------------------
    "masingar": (
        "/data/mahasiswa/romadhon/perez1/"
        "masingar/aerosol/2021/"
        "dataaerosoldecember.nc"
    ),

    # --------------------------------------------------------
    # OUTPUT TXT
    # --------------------------------------------------------
    "output_txt": (
        "/data/mahasiswa/romadhon/perez1/"
        "perez_december.txt"
    ),

    # --------------------------------------------------------
    # QC
    # --------------------------------------------------------
    "min_himawari_scene": 5,
    "max_sza": 80.0,
    "min_ssrd_wm2": 100.0,
}


# ============================================================
# 2. CPU
# ============================================================

print("=" * 70)
print(f"CPU tersedia : {N_CPU}")
print("=" * 70)


# ============================================================
# 3. LOAD HIMAWARI
# ============================================================

print("\n" + "=" * 70)
print("HIMAWARI-8")
print("=" * 70)

hima = xr.open_mfdataset(
    CONFIG["himawari"],
    engine="h5netcdf",
    combine="by_coords",
    chunks={
        "valid_time": 68
    }
)

print(hima)

print("\nVARIABLE HIMAWARI:")
print(list(hima.data_vars))


# ============================================================
# 4. LOAD ERA5 SSRD
# ============================================================

print("\n" + "=" * 70)
print("ERA5 SSRD")
print("=" * 70)

era5ssrd = xr.open_dataset(
    CONFIG["era5_ssrd"],
    engine="h5netcdf"
)

print(era5ssrd)

print("\nVARIABLE ERA5 SSRD:")
print(list(era5ssrd.data_vars))


# ============================================================
# 5. LOAD ERA5 TCWV
# ============================================================

print("\n" + "=" * 70)
print("ERA5 TCWV")
print("=" * 70)

era5tcwv = xr.open_dataset(
    CONFIG["era5_tcwv"],
    engine="h5netcdf"
)

print(era5tcwv)

print("\nVARIABLE ERA5 TCWV:")
print(list(era5tcwv.data_vars))


# ============================================================
# 6. NORMALISASI LONGITUDE TCWV
# ============================================================

print("\n" + "=" * 70)
print("CHECK LONGITUDE ERA5")
print("=" * 70)

print(
    "SSRD longitude :",
    float(era5ssrd.longitude.min()),
    "->",
    float(era5ssrd.longitude.max())
)

print(
    "TCWV longitude :",
    float(era5tcwv.longitude.min()),
    "->",
    float(era5tcwv.longitude.max())
)


era5tcwv = era5tcwv.assign_coords(
    longitude=(
        ((era5tcwv.longitude + 180) % 360) - 180
    )
)


era5tcwv = era5tcwv.sel(
    longitude=slice(110, 116.5)
)


era5tcwv = era5tcwv.sortby(
    "longitude"
)


print(
    "TCWV longitude setelah diperbaiki :",
    float(era5tcwv.longitude.min()),
    "->",
    float(era5tcwv.longitude.max())
)


# ============================================================
# 7. MERGE ERA5
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
# 8. LOAD MASINGAR
# ============================================================

print("\n" + "=" * 70)
print("MASINGAR")
print("=" * 70)

masingar_raw = xr.open_dataset(
    CONFIG["masingar"],
    engine="h5netcdf"
)

print(masingar_raw)

print("\nVARIABLE MASINGAR:")
print(list(masingar_raw.variables))


# ============================================================
# 9. KONVERSI MASINGAR
#
# FORMAT:
#   A. Flat:
#      date + time
#
#   B. Sudah grid:
#      valid_time
# ============================================================

print("\n" + "=" * 70)
print("KONVERSI WAKTU MASINGAR")
print("=" * 70)


has_date = "date" in masingar_raw.variables
has_time = "time" in masingar_raw.variables

has_valid_time = (
    "valid_time" in masingar_raw.coords
    or
    "valid_time" in masingar_raw.variables
)


# ============================================================
# 9A. MASINGAR FLAT
# ============================================================

if has_date and has_time:

    print("Format Masingar : FLAT")
    print("Membentuk valid_time dari date + time ...")

    required_columns = [
        "date",
        "time",
        "latitude",
        "longitude",
        "od500aer",
        "od550aer",
        "od870aer",
        "angstrom_exponent",
        "angstrom_turbidity"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in masingar_raw.variables
    ]

    if missing_columns:

        raise ValueError(
            "Variabel Masingar tidak lengkap: "
            + str(missing_columns)
        )


    masingar_df = (
        masingar_raw[
            required_columns
        ]
        .to_dataframe()
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    date_str = (
        masingar_df["date"]
        .astype(str)
        .str.replace(
            ".0",
            "",
            regex=False
        )
        .str.zfill(8)
    )


    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    time_num = pd.to_numeric(
        masingar_df["time"],
        errors="coerce"
    )


    time_str = (
        time_num
        .fillna(-1)
        .astype(int)
        .astype(str)
        .str.zfill(2)
    )


    # --------------------------------------------------------
    # VALID TIME
    # --------------------------------------------------------

    masingar_df["valid_time"] = pd.to_datetime(
        date_str + time_str,
        format="%Y%m%d%H",
        errors="coerce"
    )


    # --------------------------------------------------------
    # HAPUS DATE DAN TIME
    # --------------------------------------------------------

    masingar_df = masingar_df.drop(
        columns=[
            "date",
            "time"
        ]
    )


    # --------------------------------------------------------
    # HAPUS WAKTU INVALID
    # --------------------------------------------------------

    masingar_df = masingar_df[
        masingar_df["valid_time"].notna()
    ].copy()


    # --------------------------------------------------------
    # KOORDINAT
    # --------------------------------------------------------

    masingar_df["latitude"] = (
        masingar_df["latitude"]
        .astype(np.float32)
    )

    masingar_df["longitude"] = (
        masingar_df["longitude"]
        .astype(np.float32)
    )


    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    masingar_df = masingar_df.sort_values(
        [
            "valid_time",
            "latitude",
            "longitude"
        ]
    )


    # --------------------------------------------------------
    # FLAT -> GRID
    # --------------------------------------------------------

    masingar = (
        masingar_df
        .set_index(
            [
                "valid_time",
                "latitude",
                "longitude"
            ]
        )
        .to_xarray()
    )


# ============================================================
# 9B. MASINGAR SUDAH GRID
# ============================================================

elif has_valid_time:

    print("Format Masingar : SUDAH GRID")
    print("Menggunakan valid_time yang tersedia ...")

    masingar = masingar_raw.copy()


# ============================================================
# 9C. FORMAT TIDAK DIKENALI
# ============================================================

else:

    raise ValueError(
        "Format Masingar tidak dikenali. "
        "Dataset harus memiliki date+time "
        "atau valid_time."
    )


# ============================================================
# 9D. NORMALISASI MASINGAR
# ============================================================

if "valid_time" not in masingar.coords:

    if "valid_time" in masingar.variables:

        masingar = masingar.set_coords(
            "valid_time"
        )

    else:

        raise ValueError(
            "valid_time tidak ditemukan pada Masingar."
        )


masingar = masingar.assign_coords(
    valid_time=pd.to_datetime(
        masingar.valid_time.values
    )
)


masingar = masingar.sortby(
    "valid_time"
)

masingar = masingar.sortby(
    "latitude"
)

masingar = masingar.sortby(
    "longitude"
)


print("\nMASINGAR SETELAH KONVERSI:")
print(masingar)

print(
    "\nJumlah waktu Masingar:",
    len(masingar.valid_time)
)

print(
    "Jumlah latitude:",
    len(masingar.latitude)
)

print(
    "Jumlah longitude:",
    len(masingar.longitude)
)


# ============================================================
# 10. NORMALISASI DIMENSI WAKTU
# ============================================================

print("\n" + "=" * 70)
print("PENYAMAAN DIMENSI WAKTU")
print("=" * 70)


# ------------------------------------------------------------
# HIMAWARI
# ------------------------------------------------------------

if "time" in hima.dims:

    hima = hima.rename(
        {
            "time": "valid_time"
        }
    )

elif "time" in hima.coords:

    hima = hima.rename(
        {
            "time": "valid_time"
        }
    )


# ------------------------------------------------------------
# ERA5
# ------------------------------------------------------------

if "time" in era5.dims:

    era5 = era5.rename(
        {
            "time": "valid_time"
        }
    )

elif "time" in era5.coords:

    era5 = era5.rename(
        {
            "time": "valid_time"
        }
    )


# ============================================================
# 11. DATETIME
# ============================================================

hima = hima.assign_coords(
    valid_time=pd.to_datetime(
        hima.valid_time.values
    )
)


era5 = era5.assign_coords(
    valid_time=pd.to_datetime(
        era5.valid_time.values
    )
)


masingar = masingar.assign_coords(
    valid_time=pd.to_datetime(
        masingar.valid_time.values
    )
)


# ============================================================
# 12. SORT
# ============================================================

hima = hima.sortby(
    "valid_time"
)

era5 = era5.sortby(
    "valid_time"
)

masingar = masingar.sortby(
    "valid_time"
)


# ============================================================
# 13. CHECK TIME
# ============================================================

print("\n" + "=" * 70)
print("CHECK TIME")
print("=" * 70)

print(
    "Himawari timestamp :",
    len(hima.valid_time)
)

print(
    "ERA5 timestamp     :",
    len(era5.valid_time)
)

print(
    "Masingar timestamp :",
    len(masingar.valid_time)
)


print(
    "Himawari awal      :",
    hima.valid_time.values[0]
)

print(
    "Himawari akhir     :",
    hima.valid_time.values[-1]
)


print(
    "ERA5 awal          :",
    era5.valid_time.values[0]
)

print(
    "ERA5 akhir         :",
    era5.valid_time.values[-1]
)


print(
    "Masingar awal      :",
    masingar.valid_time.values[0]
)

print(
    "Masingar akhir     :",
    masingar.valid_time.values[-1]
)


# ============================================================
# 14. CHECK PERIODE
# ============================================================

hima_month = (
    pd.Timestamp(
        hima.valid_time.values[0]
    ).to_period("M")
)

era5_month = (
    pd.Timestamp(
        era5.valid_time.values[0]
    ).to_period("M")
)

masingar_month = (
    pd.Timestamp(
        masingar.valid_time.values[0]
    ).to_period("M")
)


if not (
    hima_month == era5_month == masingar_month
):

    raise ValueError(
        "\n\n"
        "PERIODE DATA TIDAK SAMA!\n"
        f"Himawari : {hima_month}\n"
        f"ERA5     : {era5_month}\n"
        f"Masingar : {masingar_month}\n\n"
        "Jangan lanjutkan sebelum file "
        "Himawari, ERA5, dan Masingar "
        "berasal dari periode yang sama."
    )


# ============================================================
# 15. TEMPORAL COVERAGE HIMAWARI
# ============================================================

print("\n" + "=" * 70)
print("COVERAGE TEMPORAL HIMAWARI")
print("=" * 70)


hima_time = pd.DatetimeIndex(
    hima.valid_time.values
)


hour_time = hima_time.floor(
    "1h"
)


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
    int(
        (
            scene_count
            < CONFIG["min_himawari_scene"]
        ).sum()
    )
)


# ============================================================
# 16. VALID HOURS
# ============================================================

valid_hours = scene_count[
    scene_count >=
    CONFIG["min_himawari_scene"]
].index


# ============================================================
# 17. TEMPORAL AVERAGING HIMAWARI
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL AVERAGING HIMAWARI")
print("=" * 70)


hima_hourly = (
    hima
    .resample(
        valid_time="1h"
    )
    .mean()
)


hima_hourly = hima_hourly.sel(
    valid_time=hima_hourly.valid_time.isin(
        valid_hours
    )
)


print(
    "Jumlah jam Himawari setelah "
    "coverage QC :",
    len(hima_hourly.valid_time)
)


# ============================================================
# 18. TARGET GRID MASINGAR
# ============================================================

print("\n" + "=" * 70)
print("TARGET GRID MASINGAR")
print("=" * 70)


target_lat = (
    masingar.latitude.values
    .astype(np.float64)
)

target_lon = (
    masingar.longitude.values
    .astype(np.float64)
)


# ------------------------------------------------------------
# LATITUDE BOUNDS
# ------------------------------------------------------------

target_lat_b = np.empty(
    len(target_lat) + 1,
    dtype=np.float64
)


target_lat_b[1:-1] = (
    target_lat[:-1]
    +
    target_lat[1:]
) / 2.0


target_lat_b[0] = (
    target_lat[0]
    -
    (
        target_lat[1]
        -
        target_lat[0]
    ) / 2.0
)


target_lat_b[-1] = (
    target_lat[-1]
    +
    (
        target_lat[-1]
        -
        target_lat[-2]
    ) / 2.0
)


# ------------------------------------------------------------
# LONGITUDE BOUNDS
# ------------------------------------------------------------

target_lon_b = np.empty(
    len(target_lon) + 1,
    dtype=np.float64
)


target_lon_b[1:-1] = (
    target_lon[:-1]
    +
    target_lon[1:]
) / 2.0


target_lon_b[0] = (
    target_lon[0]
    -
    (
        target_lon[1]
        -
        target_lon[0]
    ) / 2.0
)


target_lon_b[-1] = (
    target_lon[-1]
    +
    (
        target_lon[-1]
        -
        target_lon[-2]
    ) / 2.0
)


target_grid = xr.Dataset(
    {
        "lat": (
            ["lat"],
            target_lat
        ),

        "lon": (
            ["lon"],
            target_lon
        ),

        "lat_b": (
            ["lat_b"],
            target_lat_b
        ),

        "lon_b": (
            ["lon_b"],
            target_lon_b
        )
    }
)


print(target_grid)


# ============================================================
# 19. HIMAWARI SOURCE GRID
# ============================================================

print("\n" + "=" * 70)
print("SOURCE GRID HIMAWARI")
print("=" * 70)


hima_lat = (
    hima.latitude.values
    .astype(np.float64)
)

hima_lon = (
    hima.longitude.values
    .astype(np.float64)
)


# ------------------------------------------------------------
# LATITUDE BOUNDS
# ------------------------------------------------------------

hima_lat_b = np.empty(
    len(hima_lat) + 1,
    dtype=np.float64
)


hima_lat_b[1:-1] = (
    hima_lat[:-1]
    +
    hima_lat[1:]
) / 2.0


hima_lat_b[0] = (
    hima_lat[0]
    -
    (
        hima_lat[1]
        -
        hima_lat[0]
    ) / 2.0
)


hima_lat_b[-1] = (
    hima_lat[-1]
    +
    (
        hima_lat[-1]
        -
        hima_lat[-2]
    ) / 2.0
)


# ------------------------------------------------------------
# LONGITUDE BOUNDS
# ------------------------------------------------------------

hima_lon_b = np.empty(
    len(hima_lon) + 1,
    dtype=np.float64
)


hima_lon_b[1:-1] = (
    hima_lon[:-1]
    +
    hima_lon[1:]
) / 2.0


hima_lon_b[0] = (
    hima_lon[0]
    -
    (
        hima_lon[1]
        -
        hima_lon[0]
    ) / 2.0
)


hima_lon_b[-1] = (
    hima_lon[-1]
    +
    (
        hima_lon[-1]
        -
        hima_lon[-2]
    ) / 2.0
)


hima_source_grid = xr.Dataset(
    {
        "lat": (
            ["lat"],
            hima_lat
        ),

        "lon": (
            ["lon"],
            hima_lon
        ),

        "lat_b": (
            ["lat_b"],
            hima_lat_b
        ),

        "lon_b": (
            ["lon_b"],
            hima_lon_b
        )
    }
)


print(hima_source_grid)


# ============================================================
# 20. REGRIDDER HIMAWARI
# ============================================================

print("\n" + "=" * 70)
print("MEMBUAT REGRIDDER HIMAWARI")
print("=" * 70)


regridder_himawari = xe.Regridder(
    hima_source_grid,
    target_grid,
    method="conservative",
    periodic=False,
    reuse_weights=False
)


print(
    "✓ Regridder Himawari berhasil dibuat"
)


# ============================================================
# 21. REGRIDDING HIMAWARI
# ============================================================

print("\n" + "=" * 70)
print("REGRIDDING HIMAWARI")
print("=" * 70)


hima_regrid = xr.Dataset()


# ------------------------------------------------------------
# Variabel Himawari yang diperlukan untuk proses
#
# radiance tetap dipertahankan sementara untuk QC,
# kemudian dibuang dari output final.
#
# saa, vza, vaa tidak diperlukan.
# ------------------------------------------------------------

hima_variables = [
    "radiance",
    "reflectance",
    "sza",
    "am",
    "epsilon",
    "I0",
    "elevation"
]


for var in hima_variables:

    if var not in hima_hourly:

        continue


    print(
        "Regridding:",
        var
    )


    data = hima_hourly[var]


    data = data.rename(
        {
            "latitude": "lat",
            "longitude": "lon"
        }
    )


    result = regridder_himawari(
        data,
        skipna=True
    )


    result = result.rename(
        {
            "lat": "latitude",
            "lon": "longitude"
        }
    )


    hima_regrid[var] = result


print(
    "\n✓ Regridding Himawari selesai"
)


# ============================================================
# 22. PREPARASI GRID ERA5
# ============================================================

print("\n" + "=" * 70)
print("PREPARASI GRID ERA5")
print("=" * 70)


era5 = era5.sortby(
    "latitude"
)

era5 = era5.sortby(
    "longitude"
)


era5_lat = (
    era5.latitude.values
    .astype(np.float64)
)

era5_lon = (
    era5.longitude.values
    .astype(np.float64)
)


print(
    "ERA5 latitude :",
    era5_lat[0],
    "->",
    era5_lat[-1]
)

print(
    "ERA5 longitude:",
    era5_lon[0],
    "->",
    era5_lon[-1]
)


# ============================================================
# 23. ERA5 SOURCE GRID
# ============================================================

# ------------------------------------------------------------
# LATITUDE BOUNDS
# ------------------------------------------------------------

era5_lat_b = np.empty(
    len(era5_lat) + 1,
    dtype=np.float64
)


era5_lat_b[1:-1] = (
    era5_lat[:-1]
    +
    era5_lat[1:]
) / 2.0


era5_lat_b[0] = (
    era5_lat[0]
    -
    (
        era5_lat[1]
        -
        era5_lat[0]
    ) / 2.0
)


era5_lat_b[-1] = (
    era5_lat[-1]
    +
    (
        era5_lat[-1]
        -
        era5_lat[-2]
    ) / 2.0
)


# ------------------------------------------------------------
# LONGITUDE BOUNDS
# ------------------------------------------------------------

era5_lon_b = np.empty(
    len(era5_lon) + 1,
    dtype=np.float64
)


era5_lon_b[1:-1] = (
    era5_lon[:-1]
    +
    era5_lon[1:]
) / 2.0


era5_lon_b[0] = (
    era5_lon[0]
    -
    (
        era5_lon[1]
        -
        era5_lon[0]
    ) / 2.0
)


era5_lon_b[-1] = (
    era5_lon[-1]
    +
    (
        era5_lon[-1]
        -
        era5_lon[-2]
    ) / 2.0
)


era5_source_grid = xr.Dataset(
    {
        "lat": (
            ["lat"],
            era5_lat
        ),

        "lon": (
            ["lon"],
            era5_lon
        ),

        "lat_b": (
            ["lat_b"],
            era5_lat_b
        ),

        "lon_b": (
            ["lon_b"],
            era5_lon_b
        )
    }
)


print(era5_source_grid)


# ============================================================
# 24. REGRIDDER ERA5
# ============================================================

print("\n" + "=" * 70)
print("MEMBUAT REGRIDDER ERA5")
print("=" * 70)


regridder_era5 = xe.Regridder(
    era5_source_grid,
    target_grid,
    method="conservative",
    periodic=False,
    reuse_weights=False
)


print(
    "✓ Regridder ERA5 berhasil dibuat"
)


# ============================================================
# 25. REGRIDDING ERA5
# ============================================================

print("\n" + "=" * 70)
print("REGRIDDING ERA5")
print("=" * 70)


era5_regrid = xr.Dataset()


for var in [
    "ssrd",
    "tcwv"
]:

    if var not in era5:

        continue


    print(
        "Regridding:",
        var
    )


    data = era5[var]


    data = data.rename(
        {
            "latitude": "lat",
            "longitude": "lon"
        }
    )


    result = regridder_era5(
        data,
        skipna=True
    )


    result = result.rename(
        {
            "lat": "latitude",
            "lon": "longitude"
        }
    )


    era5_regrid[var] = result


print(
    "\n✓ Regridding ERA5 selesai"
)


# ============================================================
# 26. SET TARGET COORDINATES
# ============================================================

hima_regrid = hima_regrid.assign_coords(
    latitude=masingar.latitude,
    longitude=masingar.longitude
)


era5_regrid = era5_regrid.assign_coords(
    latitude=masingar.latitude,
    longitude=masingar.longitude
)


# ============================================================
# 27. TEMPORAL MATCHING HIMAWARI
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL MATCHING HIMAWARI")
print("=" * 70)


hima_temporal = (
    hima_regrid
    .reindex(
        valid_time=masingar.valid_time
    )
)


print(
    "Jumlah waktu Himawari setelah matching:",
    len(hima_temporal.valid_time)
)


# ============================================================
# 28. TEMPORAL MATCHING ERA5
#
# ERA5 sudah hourly.
# Tidak dilakukan interpolasi.
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL MATCHING ERA5")
print("=" * 70)


era5_temporal = (
    era5_regrid
    .reindex(
        valid_time=masingar.valid_time
    )
)


print(
    "Jumlah waktu ERA5 setelah matching:",
    len(era5_temporal.valid_time)
)


# ============================================================
# 29. MERGE SEMUA DATA
# ============================================================

print("\n" + "=" * 70)
print("MENGGABUNGKAN SEMUA DATA")
print("=" * 70)


combined = xr.merge(
    [
        masingar,
        hima_temporal,
        era5_temporal
    ],
    compat="override",
    join="outer"
)


print(combined)


# ============================================================
# 30. CONVERT TO DATAFRAME
# ============================================================

print("\n" + "=" * 70)
print("KONVERSI KE DATAFRAME")
print("=" * 70)


df = (
    combined
    .to_dataframe()
    .reset_index()
)


print(
    "Shape sebelum QC:",
    df.shape
)


# ============================================================
# 31. DROP VARIABLE TAMBAHAN ERA5
# ============================================================

df = df.drop(
    columns=[
        "number",
        "expver"
    ],
    errors="ignore"
)


# ============================================================
# 32. KONVERSI SSRD
#
# ERA5 SSRD:
#   J/m2 per jam
#
# Menjadi:
#   W/m2
#
# Nama kolom tetap:
#   ssrd
# ============================================================

if "ssrd" in df.columns:

    df["ssrd"] = (
        df["ssrd"] / 3600.0
    )


# ============================================================
# 33. JUMLAH DATA AWAL
# ============================================================

jumlah_awal = len(df)


# ============================================================
# 34. QC VALID TIME
# ============================================================

sebelum = len(df)

df = df[
    df["valid_time"].notna()
].copy()


print(
    "QC valid_time:",
    sebelum - len(df),
    "data dibuang"
)


# ============================================================
# 35. INF -> NaN
# ============================================================

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)


# ============================================================
# 36. QC RADIANCE
#
# Radiance hanya digunakan untuk QC.
# Tidak masuk output akhir.
# ============================================================

sebelum = len(df)

df = df[
    np.isfinite(df["radiance"])
    &
    (df["radiance"] > 0)
].copy()


print(
    "QC radiance > 0:",
    sebelum - len(df),
    "data dibuang"
)


# ============================================================
# 37. QC REFLECTANCE
# ============================================================

sebelum = len(df)

df = df[
    np.isfinite(df["reflectance"])
    &
    (df["reflectance"] > 0)
].copy()


print(
    "QC reflectance > 0:",
    sebelum - len(df),
    "data dibuang"
)


# ============================================================
# 38. QC SZA
# ============================================================

sebelum = len(df)

df = df[
    np.isfinite(df["sza"])
    &
    (
        df["sza"]
        < CONFIG["max_sza"]
    )
].copy()


print(
    "QC SZA < 80:",
    sebelum - len(df),
    "data dibuang"
)


# ============================================================
# 39. QC SSRD
# ============================================================

sebelum = len(df)

df = df[
    np.isfinite(df["ssrd"])
    &
    (
        df["ssrd"]
        >= CONFIG["min_ssrd_wm2"]
    )
].copy()


print(
    "QC SSRD >= 100 W/m2:",
    sebelum - len(df),
    "data dibuang"
)


# ============================================================
# 40. QC TCWV
# ============================================================

sebelum = len(df)

df = df[
    np.isfinite(df["tcwv"])
    &
    (df["tcwv"] >= 0)
].copy()


print(
    "QC TCWV >= 0:",
    sebelum - len(df),
    "data dibuang"
)


# ============================================================
# 41. QC ANGSTROM EXPONENT
# ============================================================

sebelum = len(df)

df = df[
    np.isfinite(
        df["angstrom_exponent"]
    )
].copy()


print(
    "QC Angstrom exponent finite:",
    sebelum - len(df),
    "data dibuang"
)


# ============================================================
# 42. QC ANGSTROM TURBIDITY
# ============================================================

sebelum = len(df)

df = df[
    np.isfinite(
        df["angstrom_turbidity"]
    )
].copy()


print(
    "QC Angstrom turbidity finite:",
    sebelum - len(df),
    "data dibuang"
)


# ============================================================
# 43. DROP KOLOM YANG TIDAK DIPERLUKAN
# ============================================================

df = df.drop(
    columns=[
        "radiance",
        "saa",
        "vza",
        "vaa",
        "od500aer",
        "od550aer",
        "od870aer",
        "time",
        "himawari_scene_count",
        "himawari_coverage"
    ],
    errors="ignore"
)


# ============================================================
# 44. FINAL QC SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL QC")
print("=" * 70)


print(
    "Jumlah data awal :",
    f"{jumlah_awal:,}"
)

print(
    "Jumlah data akhir:",
    f"{len(df):,}"
)

print(
    "Data terbuang    :",
    f"{jumlah_awal - len(df):,}"
)


if jumlah_awal > 0:

    print(
        "Persentase lolos :",
        f"{len(df) / jumlah_awal * 100:.2f}%"
    )


# ============================================================
# 45. SUSUNAN KOLOM FINAL
# ============================================================

final_columns = [
    "valid_time",
    "latitude",
    "longitude",
    "angstrom_exponent",
    "angstrom_turbidity",
    "reflectance",
    "sza",
    "am",
    "epsilon",
    "I0",
    "elevation",
    "tcwv",
    "ssrd"
]


final_columns = [
    col
    for col in final_columns
    if col in df.columns
]


df = df[
    final_columns
].copy()


# ============================================================
# 46. SORT FINAL
# ============================================================

df = df.sort_values(
    [
        "valid_time",
        "latitude",
        "longitude"
    ]
)


# ============================================================
# 47. ROUND NUMERIK
# ============================================================

numeric_columns = [
    col
    for col in df.columns
    if col != "valid_time"
]


df[numeric_columns] = (
    df[numeric_columns]
    .round(3)
)


# ============================================================
# 48. FORMAT DATETIME
# ============================================================

df["valid_time"] = pd.to_datetime(
    df["valid_time"]
)


# ============================================================
# 49. OUTPUT TXT
# ============================================================

print("\n" + "=" * 70)
print("MENYIMPAN OUTPUT TXT")
print("=" * 70)


df_txt = df.copy()


df_txt["valid_time"] = (
    df_txt["valid_time"]
    .dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
)


df_txt.to_csv(
    CONFIG["output_txt"],
    sep="\t",
    index=False,
    float_format="%.3f"
)


print(
    "✓ TXT berhasil disimpan:"
)

print(
    CONFIG["output_txt"]
)


# ============================================================
# 50. FINAL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("PROSES SELESAI")
print("=" * 70)


print(
    "Jumlah baris final:",
    f"{len(df):,}"
)


print(
    "Jumlah waktu final:",
    df["valid_time"].nunique()
)


print(
    "Jumlah latitude final:",
    df["latitude"].nunique()
)


print(
    "Jumlah longitude final:",
    df["longitude"].nunique()
)


print("\nKolom final:")

for col in df.columns:

    print(
        " -",
        col
    )


print("\nOutput TXT:")

print(
    CONFIG["output_txt"]
)

print("=" * 70)