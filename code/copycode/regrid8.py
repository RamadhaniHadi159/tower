#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import pvlib

# ============================================================
# 1. CONFIGURATION
# ============================================================

N_CPU = 40

CONFIG = {
    "himawari": (
        "/data/mahasiswa/romadhon/perez1/"
        "himawari/2021/himawari_december/nc/"
        "himawari_*.nc"
    ),
    "era5_ssrd": (
        "/data/mahasiswa/romadhon/perez1/"
        "era5/ssrd/2021/"
        "dataera5ssrddecember.nc"
    ),
    "era5_tcwv": (
        "/data/mahasiswa/romadhon/perez1/"
        "era5/tcwv/2021/"
        "dataera5tcwvdecember.nc"
    ),
    "masingar": (
        "/data/mahasiswa/romadhon/perez1/"
        "masingar/aerosol/2021/nc/"
        "dataaerosoldecember.nc"
    ),
    "output_txt": (
        "/data/mahasiswa/romadhon/perez1/perez/"
        "perez_december.txt"
    ),
    "min_himawari_scene": 5,
    "max_sza": 80.0,
    "min_ssrd_wm2": 100.0,
}

print("=" * 70)
print(f"CPU tersedia : {N_CPU}")
print("=" * 70)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def normalize_time_name(ds):
    """Ubah dimensi/koordinat time menjadi valid_time bila diperlukan."""
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        return ds

    if "time" in ds.dims or "time" in ds.coords:
        return ds.rename({"time": "valid_time"})

    raise ValueError("Dataset tidak memiliki time atau valid_time.")


def normalize_datetime(ds):
    ds = normalize_time_name(ds)
    ds = ds.assign_coords(
        valid_time=pd.to_datetime(ds.valid_time.values)
    )
    return ds.sortby("valid_time")


def normalize_longitude(ds):
    """Normalisasi longitude menjadi -180..180 dan urut menaik."""
    if "longitude" not in ds.coords:
        raise ValueError("Koordinat longitude tidak ditemukan.")

    ds = ds.assign_coords(
        longitude=((ds.longitude + 180) % 360) - 180
    )
    return ds.sortby("longitude")


def make_bounds(coord):
    """Membuat cell bounds 1D dari koordinat pusat grid."""
    coord = np.asarray(coord, dtype=np.float64)


    if coord.ndim != 1 or len(coord) < 2:
        raise ValueError("Koordinat grid harus 1D dan minimal 2 titik.")

    if not np.all(np.diff(coord) > 0):
        raise ValueError("Koordinat grid harus monoton naik.")

    bounds = np.empty(len(coord) + 1, dtype=np.float64)

    bounds[1:-1] = (coord[:-1] + coord[1:]) / 2.0
    bounds[0] = coord[0] - (coord[1] - coord[0]) / 2.0
    bounds[-1] = coord[-1] + (coord[-1] - coord[-2]) / 2.0

    return bounds


def make_xesmf_grid(lat, lon):
    """Grid xESMF untuk conservative regridding."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    return xr.Dataset(
        {
            "lat": (["lat"], lat),
            "lon": (["lon"], lon),
            "lat_b": (["lat_b"], make_bounds(lat)),
            "lon_b": (["lon_b"], make_bounds(lon)),
        }
    )


def regrid_variable(regridder, data):
    """Regrid satu DataArray latitude/longitude ke target grid."""
    data = data.rename(
        {
            "latitude": "lat",
            "longitude": "lon",
        }
    )

    result = regridder(
        data,
        skipna=True
    )

    return result.rename(
        {
            "lat": "latitude",
            "lon": "longitude",
        }
    )


def exact_time_report(source_time, target_time, name):
    """Audit kecocokan timestamp exact sebelum reindex."""
    source = pd.DatetimeIndex(pd.to_datetime(source_time))
    target = pd.DatetimeIndex(pd.to_datetime(target_time))

    common = source.intersection(target)
    missing = target.difference(source)

    print(f"\n{name}")
    print(f"  Timestamp sumber : {len(source)}")
    print(f"  Timestamp target : {len(target)}")
    print(f"  Exact match      : {len(common)}")
    print(f"  Tidak ditemukan : {len(missing)}")

    if len(missing) > 0:
        print("  Contoh missing   :", list(missing[:5]))

    return common, missing


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
    chunks={"valid_time": 68},
)

hima = normalize_datetime(hima)

# Pastikan koordinat spasial monoton naik sebelum bounds dibuat
hima = hima.sortby("latitude")
hima = hima.sortby("longitude")

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
    engine="h5netcdf",
)

era5ssrd = normalize_datetime(era5ssrd)
era5ssrd = normalize_longitude(era5ssrd)
era5ssrd = era5ssrd.sortby("latitude")

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
    engine="h5netcdf",
)

era5tcwv = normalize_datetime(era5tcwv)
era5tcwv = normalize_longitude(era5tcwv)
era5tcwv = era5tcwv.sortby("latitude")

print(era5tcwv)
print("\nVARIABLE ERA5 TCWV:")
print(list(era5tcwv.data_vars))


# ============================================================
# 6. LOAD MASINGAR
# ============================================================

print("\n" + "=" * 70)
print("MASINGAR")
print("=" * 70)

masingar_raw = xr.open_dataset(
    CONFIG["masingar"],
    engine="h5netcdf",
)

print(masingar_raw)
print("\nVARIABLE MASINGAR:")
print(list(masingar_raw.variables))


# ============================================================
# 7. KONVERSI WAKTU MASINGAR
# ============================================================

print("\n" + "=" * 70)
print("KONVERSI WAKTU MASINGAR")
print("=" * 70)

has_date = "date" in masingar_raw.variables
has_time = "time" in masingar_raw.variables

has_valid_time = (
    "valid_time" in masingar_raw.coords
    or "valid_time" in masingar_raw.variables
)

if has_date and has_time:

    print("Format Masingar : FLAT")

    required_columns = [
        "date",
        "time",
        "latitude",
        "longitude",
        "od500aer",
        "od550aer",
        "od870aer",
        "angstrom_turbidity",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in masingar_raw.variables
    ]

    if missing_columns:
        raise ValueError(
            "Variabel Masingar tidak lengkap: "
            + str(missing_columns)
        )

    masingar_df = (
        masingar_raw[required_columns]
        .to_dataframe()
        .reset_index(drop=True)
    )

    date_str = (
        masingar_df["date"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.zfill(8)
    )

    time_num = pd.to_numeric(
        masingar_df["time"],
        errors="coerce",
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
        errors="coerce",
    )

    masingar_df = masingar_df.drop(
        columns=["date", "time"]
    )

    masingar_df = masingar_df[
        masingar_df["valid_time"].notna()
    ].copy()

    masingar_df["latitude"] = (
        masingar_df["latitude"].astype(np.float64)
    )

    masingar_df["longitude"] = (
        masingar_df["longitude"].astype(np.float64)
    )

    masingar_df = masingar_df.sort_values(
        ["valid_time", "latitude", "longitude"]
    )

    # Cegah duplikasi index yang dapat membuat to_xarray bermasalah
    duplicated = masingar_df.duplicated(
        subset=["valid_time", "latitude", "longitude"]
    )

    if duplicated.any():
        raise ValueError(
            "Ditemukan duplikasi kombinasi "
            "valid_time-latitude-longitude pada Masingar."
        )

    masingar = (
        masingar_df
        .set_index(
            ["valid_time", "latitude", "longitude"]
        )
        .to_xarray()
    )

elif has_valid_time:

    print("Format Masingar : SUDAH GRID")
    masingar = masingar_raw.copy()

else:

    raise ValueError(
        "Format Masingar tidak dikenali. "
        "Dataset harus memiliki date+time atau valid_time."
    )


if "valid_time" not in masingar.coords:
    if "valid_time" in masingar.variables:
        masingar = masingar.set_coords("valid_time")
    else:
        raise ValueError(
            "valid_time tidak ditemukan pada Masingar."
        )

masingar = masingar.assign_coords(
    valid_time=pd.to_datetime(masingar.valid_time.values)
)

masingar = masingar.sortby("valid_time")
masingar = masingar.sortby("latitude")
masingar = masingar.sortby("longitude")

print("\nMASINGAR SETELAH KONVERSI:")
print(masingar)

print("\nJumlah waktu Masingar:", len(masingar.valid_time))
print("Jumlah latitude:", len(masingar.latitude))
print("Jumlah longitude:", len(masingar.longitude))


# ============================================================
# 8. CHECK PERIODE
# ============================================================

print("\n" + "=" * 70)
print("CHECK TIME DAN PERIODE")
print("=" * 70)

print("Himawari awal  :", hima.valid_time.values[0])
print("Himawari akhir :", hima.valid_time.values[-1])

print("ERA5 SSRD awal :", era5ssrd.valid_time.values[0])
print("ERA5 SSRD akhir:", era5ssrd.valid_time.values[-1])

print("ERA5 TCWV awal :", era5tcwv.valid_time.values[0])
print("ERA5 TCWV akhir:", era5tcwv.valid_time.values[-1])

print("Masingar awal  :", masingar.valid_time.values[0])
print("Masingar akhir :", masingar.valid_time.values[-1])

hima_month = pd.Timestamp(
    hima.valid_time.values[0]
).to_period("M")

ssrd_month = pd.Timestamp(
    era5ssrd.valid_time.values[0]
).to_period("M")

tcwv_month = pd.Timestamp(
    era5tcwv.valid_time.values[0]
).to_period("M")

masingar_month = pd.Timestamp(
    masingar.valid_time.values[0]
).to_period("M")

if not (
    hima_month
    == ssrd_month
    == tcwv_month
    == masingar_month
):
    raise ValueError(
        "\nPERIODE DATA TIDAK SAMA!\n"
        f"Himawari : {hima_month}\n"
        f"SSRD     : {ssrd_month}\n"
        f"TCWV     : {tcwv_month}\n"
        f"Masingar : {masingar_month}\n"
    )


# ============================================================
# 9. TEMPORAL COVERAGE HIMAWARI
# ============================================================

print("\n" + "=" * 70)
print("COVERAGE TEMPORAL HIMAWARI")
print("=" * 70)

hima_time = pd.DatetimeIndex(
    hima.valid_time.values
)

hour_time = hima_time.floor("1h")

scene_count = (
    pd.Series(1, index=hour_time)
    .groupby(level=0)
    .sum()
)

print("\nDistribusi jumlah scene per jam:")
print(
    scene_count
    .value_counts()
    .sort_index()
)

print("\nJumlah jam 6/6 :", int((scene_count == 6).sum()))
print("Jumlah jam 5/6 :", int((scene_count == 5).sum()))
print(
    "Jumlah jam <5/6:",
    int(
        (
            scene_count
            < CONFIG["min_himawari_scene"]
        ).sum()
    ),
)

valid_hours = scene_count[
    scene_count >= CONFIG["min_himawari_scene"]
].index


# ============================================================
# 10. TEMPORAL AVERAGING HIMAWARI
#
# Hanya variabel observasi yang memang perlu diagregasi.
# SZA, AM, epsilon dan I0 TIDAK dirata-ratakan di sini.
# Elevation bersifat statis dan ditangani terpisah.
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL AVERAGING HIMAWARI")
print("=" * 70)

hima_obs_variables = [
    var for var in ["radiance", "reflectance"]
    if var in hima.data_vars
]

if "reflectance" not in hima_obs_variables:
    raise ValueError(
        "Variabel reflectance tidak ditemukan pada Himawari."
    )

hima_hourly = (
    hima[hima_obs_variables]
    .resample(valid_time="1h")
    .mean(skipna=True)
)

hima_hourly = hima_hourly.sel(
    valid_time=hima_hourly.valid_time.isin(
        valid_hours
    )
)

print(
    "Jumlah jam Himawari setelah coverage QC:",
    len(hima_hourly.valid_time),
)


# ============================================================
# 11. TARGET GRID MASINGAR
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

target_grid = make_xesmf_grid(
    target_lat,
    target_lon,
)

print(target_grid)


# ============================================================
# 12. SOURCE GRID HIMAWARI
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

hima_source_grid = make_xesmf_grid(
    hima_lat,
    hima_lon,
)

print(hima_source_grid)


# ============================================================
# 13. REGRIDDER HIMAWARI
# ============================================================

print("\n" + "=" * 70)
print("MEMBUAT REGRIDDER HIMAWARI")
print("=" * 70)

regridder_himawari = xe.Regridder(
    hima_source_grid,
    target_grid,
    method="conservative",
    periodic=False,
    reuse_weights=False,
)

print("✓ Regridder Himawari berhasil dibuat")


# ============================================================
# 14. REGRIDDING HIMAWARI
#
# Hanya radiance dan reflectance yang diagregasi dari citra.
# ============================================================

print("\n" + "=" * 70)
print("REGRIDDING HIMAWARI")
print("=" * 70)

hima_regrid = xr.Dataset()

for var in hima_obs_variables:

    print("Regridding:", var)

    hima_regrid[var] = regrid_variable(
        regridder_himawari,
        hima_hourly[var],
    )


# ============================================================
# 15. REGRIDDING ELEVATION
#
# Elevation statis: tidak perlu temporal averaging.
# Diambil dari satu time slice lalu dibuat area-weighted mean
# pada grid Masingar.
# ============================================================

if "elevation" in hima.data_vars:

    print("Regridding: elevation")

    elevation_source = hima["elevation"]

    if "valid_time" in elevation_source.dims:
        elevation_source = elevation_source.isel(
            valid_time=0,
            drop=True,
        )

    hima_regrid["elevation"] = regrid_variable(
        regridder_himawari,
        elevation_source,
    )


print("\n✓ Regridding Himawari selesai")


# ============================================================
# 16. PREPARASI DAN REGRIDDING ERA5 SSRD
#
# SSRD dan TCWV diregrid dari grid masing-masing.
# Tidak digabungkan sebelum regridding agar perbedaan koordinat
# sumber tidak tersembunyi oleh merge outer.
# ============================================================

print("\n" + "=" * 70)
print("REGRIDDING ERA5 SSRD")
print("=" * 70)

if "ssrd" not in era5ssrd.data_vars:
    raise ValueError("Variabel ssrd tidak ditemukan.")

ssrd_source_grid = make_xesmf_grid(
    era5ssrd.latitude.values.astype(np.float64),
    era5ssrd.longitude.values.astype(np.float64),
)

regridder_ssrd = xe.Regridder(
    ssrd_source_grid,
    target_grid,
    method="conservative",
    periodic=False,
    reuse_weights=False,
)

era5_ssrd_regrid = xr.Dataset(
    {
        "ssrd": regrid_variable(
            regridder_ssrd,
            era5ssrd["ssrd"],
        )
    }
)

print("✓ Regridding SSRD selesai")


# ============================================================
# 17. PREPARASI DAN REGRIDDING ERA5 TCWV
# ============================================================

print("\n" + "=" * 70)
print("REGRIDDING ERA5 TCWV")
print("=" * 70)

if "tcwv" not in era5tcwv.data_vars:
    raise ValueError("Variabel tcwv tidak ditemukan.")

tcwv_source_grid = make_xesmf_grid(
    era5tcwv.latitude.values.astype(np.float64),
    era5tcwv.longitude.values.astype(np.float64),
)

regridder_tcwv = xe.Regridder(
    tcwv_source_grid,
    target_grid,
    method="conservative",
    periodic=False,
    reuse_weights=False,
)

era5_tcwv_regrid = xr.Dataset(
    {
        "tcwv": regrid_variable(
            regridder_tcwv,
            era5tcwv["tcwv"],
        )
    }
)

print("✓ Regridding TCWV selesai")


# ============================================================
# 18. SET TARGET COORDINATES
# ============================================================

for ds in [
    hima_regrid,
    era5_ssrd_regrid,
    era5_tcwv_regrid,
]:
    ds.coords["latitude"] = masingar.latitude
    ds.coords["longitude"] = masingar.longitude


# ============================================================
# 19. AUDIT + EXACT TEMPORAL MATCHING
# ============================================================

print("\n" + "=" * 70)
print("AUDIT EXACT TEMPORAL MATCHING")
print("=" * 70)

exact_time_report(
    hima_regrid.valid_time.values,
    masingar.valid_time.values,
    "HIMAWARI vs MASINGAR",
)

exact_time_report(
    era5_ssrd_regrid.valid_time.values,
    masingar.valid_time.values,
    "ERA5 SSRD vs MASINGAR",
)

exact_time_report(
    era5_tcwv_regrid.valid_time.values,
    masingar.valid_time.values,
    "ERA5 TCWV vs MASINGAR",
)

# Exact matching tetap dipertahankan.
hima_temporal = hima_regrid.reindex(
    valid_time=masingar.valid_time
)

ssrd_temporal = era5_ssrd_regrid.reindex(
    valid_time=masingar.valid_time
)

tcwv_temporal = era5_tcwv_regrid.reindex(
    valid_time=masingar.valid_time
)


# ============================================================
# 20. MERGE SEMUA DATA
#
# join="exact" dipakai karena setelah reindex dan regrid,
# seluruh dataset memang seharusnya mempunyai koordinat target
# yang sama. Jika tidak sama, program berhenti dan masalah
# koordinat terlihat jelas.
# ============================================================

print("\n" + "=" * 70)
print("MENGGABUNGKAN SEMUA DATA")
print("=" * 70)

combined = xr.merge(
    [
        masingar,
        hima_temporal,
        ssrd_temporal,
        tcwv_temporal,
    ],
    compat="override",
    join="exact",
)

print(combined)


# ============================================================
# 21. CONVERT TO DATAFRAME
# ============================================================

print("\n" + "=" * 70)
print("KONVERSI KE DATAFRAME")
print("=" * 70)

df = (
    combined
    .to_dataframe()
    .reset_index()
)

print("Shape sebelum parameter target:", df.shape)

df = df.drop(
    columns=["number", "expver"],
    errors="ignore",
)


# ============================================================
# 22. HITUNG ULANG SZA PADA GRID/WAKTU TARGET
#
# Catatan:
# valid_time diasumsikan UTC. Naive datetime diberikan ke pvlib
# sebagai UTC melalui lokalisasi sementara di bagian ini.
# ============================================================

print("\n" + "=" * 70)
print("MENGHITUNG SZA PADA GRID TARGET")
print("=" * 70)

time_utc = pd.DatetimeIndex(
    pd.to_datetime(df["valid_time"])
)

if time_utc.tz is None:
    time_utc = time_utc.tz_localize("UTC")
else:
    time_utc = time_utc.tz_convert("UTC")

solar = pvlib.solarposition.get_solarposition(
    time=time_utc,
    latitude=df["latitude"].to_numpy(dtype=float),
    longitude=df["longitude"].to_numpy(dtype=float),
)

df["sza"] = solar["zenith"].to_numpy(dtype=float)


# ============================================================
# 23. HITUNG ULANG AIR MASS DARI SZA TARGET
# ============================================================

df["am"] = (
    pvlib.atmosphere
    .get_relative_airmass(
        df["sza"].to_numpy(dtype=float),
        model="kastenyoung1989",
    )
    .astype(float)
)

df.loc[
    ~np.isfinite(df["am"]),
    "am",
] = np.nan


# ============================================================
# 24. HITUNG ULANG EPSILON DAN I0 DARI WAKTU TARGET
#
# Formula dipertahankan sama dengan kode perhitungan awal.
# ============================================================

julianday = (
    pd.to_datetime(df["valid_time"])
    .dt.dayofyear
    .to_numpy(dtype=float)
)

df["epsilon"] = (
    1.0
    + 0.033
    * np.cos(
        2.0
        * np.pi
        * julianday
        / 365.0
    )
)

df["I0"] = 1361.0 * df["epsilon"]


# ============================================================
# 25. KONVERSI SSRD
#
# Hanya benar bila ssrd adalah akumulasi energi J/m² selama
# interval 1 jam. Verifikasi metadata file ERA5 Anda.
# ============================================================

if "ssrd" in df.columns:
    df["ssrd"] = df["ssrd"] / 3600.0


# ============================================================
# 26. KONVERSI TCWV
#
# kg/m² = mm air; 10 mm = 1 cm.
# ============================================================

if "tcwv" in df.columns:
    df["tcwv"] = df["tcwv"] / 10.0


# ============================================================
# 27. JUMLAH DATA AWAL
# ============================================================

jumlah_awal = len(df)


# ============================================================
# 28. QC
# ============================================================

sebelum = len(df)
df = df[df["valid_time"].notna()].copy()
print(
    "QC valid_time:",
    sebelum - len(df),
    "data dibuang",
)

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True,
)


# Radiance hanya untuk QC.
if "radiance" in df.columns:
    sebelum = len(df)

    df = df[
        np.isfinite(df["radiance"])
        & (df["radiance"] > 0)
    ].copy()

    print(
        "QC radiance > 0:",
        sebelum - len(df),
        "data dibuang",
    )


sebelum = len(df)

df = df[
    np.isfinite(df["reflectance"])
    & (df["reflectance"] > 0)
].copy()

print(
    "QC reflectance > 0:",
    sebelum - len(df),
    "data dibuang",
)


sebelum = len(df)

df = df[
    np.isfinite(df["sza"])
    & (df["sza"] < CONFIG["max_sza"])
].copy()

print(
    "QC SZA < 80:",
    sebelum - len(df),
    "data dibuang",
)


sebelum = len(df)

df = df[
    np.isfinite(df["ssrd"])
    & (
        df["ssrd"]
        >= CONFIG["min_ssrd_wm2"]
    )
].copy()

print(
    "QC SSRD >= 100 W/m²:",
    sebelum - len(df),
    "data dibuang",
)


sebelum = len(df)

df = df[
    np.isfinite(df["tcwv"])
    & (df["tcwv"] >= 0)
].copy()

print(
    "QC TCWV >= 0 cm:",
    sebelum - len(df),
    "data dibuang",
)


sebelum = len(df)

df = df[
    np.isfinite(
        df["angstrom_turbidity"]
    )
].copy()

print(
    "QC Angstrom turbidity finite:",
    sebelum - len(df),
    "data dibuang",
)


# ============================================================
# 29. DROP KOLOM YANG TIDAK DIPERLUKAN
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
        "himawari_coverage",
        "angstrom_exponent",
    ],
    errors="ignore",
)


# ============================================================
# 30. FINAL QC SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL QC")
print("=" * 70)

print("Jumlah data awal :", f"{jumlah_awal:,}")
print("Jumlah data akhir:", f"{len(df):,}")
print(
    "Data terbuang    :",
    f"{jumlah_awal - len(df):,}",
)

if jumlah_awal > 0:
    print(
        "Persentase lolos :",
        f"{len(df) / jumlah_awal * 100:.2f}%",
    )


# ============================================================
# 31. RENAME KOLOM DENGAN SATUAN
# ============================================================

rename_columns = {
    "angstrom_turbidity":
        "angstrom_turbidity(dimensionless)",
    "reflectance":
        "reflectance(dimensionless)",
    "sza":
        "sza(°)",
    "am":
        "am(dimensionless)",
    "epsilon":
        "epsilon(dimensionless)",
    "I0":
        "I0(W/m²)",
    "elevation":
        "elevation(m)",
    "tcwv":
        "tcwv(cm)",
    "ssrd":
        "ssrd(W/m²)",
}

df = df.rename(
    columns=rename_columns
)


# ============================================================
# 32. SUSUNAN KOLOM FINAL
# ============================================================

final_columns = [
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
]

missing_final = [
    col for col in final_columns
    if col not in df.columns
]

if missing_final:
    raise ValueError(
        "Kolom final tidak lengkap: "
        + str(missing_final)
    )

df = df[final_columns].copy()


# ============================================================
# 33. SORT + ROUND
# ============================================================

df = df.sort_values(
    ["valid_time", "latitude", "longitude"]
)

numeric_columns = [
    col for col in df.columns
    if col != "valid_time"
]

df[numeric_columns] = (
    df[numeric_columns]
    .round(3)
)

df["valid_time"] = pd.to_datetime(
    df["valid_time"]
)


# ============================================================
# 34. OUTPUT TXT
# ============================================================

print("\n" + "=" * 70)
print("MENYIMPAN OUTPUT TXT")
print("=" * 70)

df_txt = df.copy()

df_txt["valid_time"] = (
    df_txt["valid_time"]
    .dt.strftime("%Y-%m-%d %H:%M:%S")
)

df_txt.to_csv(
    CONFIG["output_txt"],
    sep="\t",
    index=False,
    float_format="%.3f",
)

print("✓ TXT berhasil disimpan:")
print(CONFIG["output_txt"])


# ============================================================
# 35. FINAL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("PROSES SELESAI")
print("=" * 70)

print("Jumlah baris final:", f"{len(df):,}")
print(
    "Jumlah waktu final:",
    df["valid_time"].nunique(),
)
print(
    "Jumlah latitude final:",
    df["latitude"].nunique(),
)
print(
    "Jumlah longitude final:",
    df["longitude"].nunique(),
)

print("\nKolom final:")
for col in df.columns:
    print(" -", col)

print("\nOutput TXT:")
print(CONFIG["output_txt"])
print("=" * 70)
