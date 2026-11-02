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
#   Coverage minimum 5/6 scene per jam
#
# TARGET GRID:
#   Masingar
#   12 latitude x 17 longitude
#   744 hourly timestamps
#
# QC:
#   Radiance > 0
#   Reflectance > 0
#   SZA < 80 deg
#   SSRD >= 100 W/m2
#   TCWV >= 0
#   AOD > 0
#   Angstrom exponent finite
#   Angstrom turbidity finite
#
# TIDAK ADA:
#   Coordinate bounds QC
#   Interpolasi missing data
#   Recalculation alpha/beta
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
        "integrasi_december2021.txt"
    ),

    # --------------------------------------------------------
    # OUTPUT NETCDF
    # --------------------------------------------------------
    "output_nc": (
        "/data/mahasiswa/romadhon/perez1/"
        "integrasi_december2021.nc"
    ),

    # --------------------------------------------------------
    # PARAMETER QC
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
# Normalisasi longitude TCWV
# ------------------------------------------------------------

era5tcwv = era5tcwv.assign_coords(
    longitude=(
        ((era5tcwv.longitude + 180) % 360) - 180
    )
)


# ------------------------------------------------------------
# Crop wilayah
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
# ROBUST:
#   A. Jika masih flat:
#      date + time + latitude + longitude
#
#   B. Jika sudah memiliki valid_time:
#      langsung digunakan
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
# 9A. MASINGAR MASIH FLAT
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
    # date = YYYYMMDD
    # time = HH
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


    masingar_df["valid_time"] = pd.to_datetime(
        date_str + time_str,
        format="%Y%m%d%H",
        errors="coerce"
    )


    # --------------------------------------------------------
    # Hapus date dan time asli
    # --------------------------------------------------------

    masingar_df = masingar_df.drop(
        columns=[
            "date",
            "time"
        ]
    )


    # --------------------------------------------------------
    # Hapus waktu invalid
    # --------------------------------------------------------

    masingar_df = masingar_df[
        masingar_df["valid_time"].notna()
    ].copy()


    # --------------------------------------------------------
    # Tipe koordinat
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
    # Sort
    # --------------------------------------------------------

    masingar_df = masingar_df.sort_values(
        [
            "valid_time",
            "latitude",
            "longitude"
        ]
    )


    # --------------------------------------------------------
    # Flat -> xarray grid
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
# Himawari
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
# 12. SORT DATASET
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
# 14. CEK PERIODE
#
# Jangan sampai Desember Himawari digabung
# dengan Januari ERA5/Masingar.
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
# 15. CHECK TARGET TIME MASINGAR
# ============================================================

if len(masingar.valid_time) == 0:

    raise ValueError(
        "Masingar tidak memiliki timestamp."
    )


# ============================================================
# 16. TEMPORAL COVERAGE HIMAWARI
# ============================================================

print("\n" + "=" * 70)
print("COVERAGE TEMPORAL HIMAWARI")
print("=" * 70)


hima_time = pd.DatetimeIndex(
    hima.valid_time.values
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
    int(
        (
            scene_count
            < CONFIG["min_himawari_scene"]
        ).sum()
    )
)


# ============================================================
# 17. VALID HOURS
# ============================================================

valid_hours = scene_count[
    scene_count >=
    CONFIG["min_himawari_scene"]
].index


# ============================================================
# 18. TEMPORAL AVERAGING HIMAWARI
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


# ------------------------------------------------------------
# Hanya jam 5/6 dan 6/6
# ------------------------------------------------------------

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


print(
    "Timestamp Himawari setelah "
    "temporal averaging :",
    len(hima_hourly.valid_time)
)


# ============================================================
# 19. BUAT TARGET GRID MASINGAR
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
# Latitude bounds
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
# Longitude bounds
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
# 20. HIMAWARI SOURCE GRID
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
# Latitude bounds
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
# Longitude bounds
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
# 21. REGRIDDER HIMAWARI
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
# 22. REGRIDDING HIMAWARI
# ============================================================

print("\n" + "=" * 70)
print("REGRIDDING HIMAWARI")
print("=" * 70)


hima_regrid = xr.Dataset()


hima_variables = [
    "radiance",
    "reflectance",
    "sza",
    "saa",
    "vza",
    "vaa",
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

print(hima_regrid)


# ============================================================
# 23. ERA5 SORT LATITUDE
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
    "→",
    era5_lat[-1]
)

print(
    "ERA5 longitude:",
    era5_lon[0],
    "→",
    era5_lon[-1]
)


# ============================================================
# 24. ERA5 SOURCE GRID
# ============================================================

# ------------------------------------------------------------
# Latitude bounds
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
# Longitude bounds
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
# 25. REGRIDDER ERA5
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
# 26. REGRIDDING ERA5
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

print(era5_regrid)


# ============================================================
# 27. SET TARGET COORDINATES
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
# 28. TEMPORAL MATCHING HIMAWARI
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
# 29. TEMPORAL MATCHING ERA5
# ============================================================
#
# ERA5 sudah hourly.
# Tidak dilakukan interpolasi.
#
# Exact matching ke waktu Masingar.
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
# 30. MERGE SEMUA DATASET
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
# 31. CONVERT TO DATAFRAME
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
# 32. DROP VARIABLE TAMBAHAN ERA5
# ============================================================

df = df.drop(
    columns=[
        "number",
        "expver"
    ],
    errors="ignore"
)


# ============================================================
# 33. KONVERSI SSRD
# ============================================================

if "ssrd" in df.columns:

    df["ssrd_wm2"] = (
        df["ssrd"] / 3600.0
    )


# ============================================================
# 34. JUMLAH DATA AWAL
# ============================================================

jumlah_awal = len(df)


# ============================================================
# 35. QC VALID TIME
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
# 36. INF -> NaN
# ============================================================

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)


# ============================================================
# 37. QC RADIANCE
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
# 38. QC REFLECTANCE
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
# 39. QC SZA
# ============================================================

sebelum = len(df)

df = df[
    np.isfinite(df["sza"])
    &
    (df["sza"] < CONFIG["max_sza"])
].copy()


print(
    "QC SZA < 80:",
    sebelum - len(df),
    "data dibuang"
)


# ============================================================
# 40. QC SSRD
# ============================================================

sebelum = len(df)

df = df[
    np.isfinite(df["ssrd_wm2"])
    &
    (
        df["ssrd_wm2"]
        >= CONFIG["min_ssrd_wm2"]
    )
].copy()


print(
    "QC SSRD >= 100 W/m2:",
    sebelum - len(df),
    "data dibuang"
)


# ============================================================
# 41. QC TCWV
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
# 42. QC AOD
# ============================================================

for col in [
    "od500aer",
    "od550aer",
    "od870aer"
]:

    if col not in df.columns:

        continue


    sebelum = len(df)

    df = df[
        np.isfinite(df[col])
        &
        (df[col] > 0)
    ].copy()


    print(
        f"QC {col} > 0:",
        sebelum - len(df),
        "data dibuang"
    )


# ============================================================
# 43. QC ANGSTROM EXPONENT
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
# 44. QC ANGSTROM TURBIDITY
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
# 45. FINAL QC SUMMARY
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
# 46. DROP KOLOM YANG TIDAK DIPERLUKAN
# ============================================================

df = df.drop(
    columns=[
        "time",
        "od500aer",
        "od550aer",
        "od870aer",
        "himawari_scene_count",
        "himawari_coverage"
    ],
    errors="ignore"
)


# ============================================================
# 47. SUSUNAN KOLOM FINAL
# ============================================================

final_columns = [
    "valid_time",
    "latitude",
    "longitude",
    "angstrom_exponent",
    "angstrom_turbidity",
    "radiance",
    "reflectance",
    "sza",
    "saa",
    "vza",
    "vaa",
    "am",
    "epsilon",
    "I0",
    "elevation",
    "tcwv",
    "ssrd_wm2"
]


# ------------------------------------------------------------
# Hanya ambil kolom yang tersedia
# ------------------------------------------------------------

final_columns = [
    col
    for col in final_columns
    if col in df.columns
]


df = df[
    final_columns
].copy()


# ============================================================
# 48. SORT FINAL
# ============================================================

df = df.sort_values(
    [
        "valid_time",
        "latitude",
        "longitude"
    ]
)


# ============================================================
# 49. ROUND NUMERIK
# ============================================================

numeric_columns = [
    col
    for col in df.columns
    if col not in [
        "valid_time"
    ]
]


df[numeric_columns] = (
    df[numeric_columns]
    .round(3)
)


# ============================================================
# 50. FORMAT VALID TIME
# ============================================================

df["valid_time"] = pd.to_datetime(
    df["valid_time"]
)


# ============================================================
# 51. OUTPUT TXT
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
# 52. OUTPUT NETCDF
# ============================================================

print("\n" + "=" * 70)
print("MENYIMPAN OUTPUT NETCDF")
print("=" * 70)


# ------------------------------------------------------------
# valid_time tetap datetime64
# ------------------------------------------------------------

df_nc = df.copy()


# ------------------------------------------------------------
# Set index
# ------------------------------------------------------------

df_nc = df_nc.set_index(
    [
        "valid_time",
        "latitude",
        "longitude"
    ]
)


# ------------------------------------------------------------
# DataFrame -> xarray
# ------------------------------------------------------------

ds_output = xr.Dataset.from_dataframe(
    df_nc
)


# ------------------------------------------------------------
# Sort
# ------------------------------------------------------------

ds_output = ds_output.sortby(
    "valid_time"
)

ds_output = ds_output.sortby(
    "latitude"
)

ds_output = ds_output.sortby(
    "longitude"
)


# ------------------------------------------------------------
# Attribute
# ------------------------------------------------------------

ds_output.attrs[
    "description"
] = (
    "Integrated Himawari-8, ERA5 SSRD, "
    "ERA5 TCWV, and Masingar aerosol data"
)

ds_output.attrs[
    "period"
] = "December 2021"

ds_output.attrs[
    "spatial_regridding"
] = (
    "Conservative / area-weighted "
    "using xESMF"
)

ds_output.attrs[
    "himawari_temporal_resolution"
] = (
    "10-minute aggregated to hourly"
)

ds_output.attrs[
    "himawari_coverage"
] = (
    "Minimum 5 of 6 scenes per hour"
)


# ------------------------------------------------------------
# Simpan NetCDF
# ------------------------------------------------------------

ds_output.to_netcdf(
    CONFIG["output_nc"],
    engine="h5netcdf"
)


print(
    "✓ NetCDF berhasil disimpan:"
)

print(
    CONFIG["output_nc"]
)


# ============================================================
# 53. FINAL INFORMATION
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


print(
    "\nKolom final:"
)

print(
    list(df.columns)
)


print("\nOutput:")

print(
    "TXT :",
    CONFIG["output_txt"]
)

print(
    "NC  :",
    CONFIG["output_nc"]
)

print("=" * 70)