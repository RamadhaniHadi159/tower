import calendar
import glob
import os
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import pvlib


# ============================================================
# KONFIGURASI INPUT & OUTPUT (DAPAT DIUBAH DI SINI)
# ============================================================

# Daftar 3 Bulan yang Diproses (1 Musim)
# Contoh pilihan musim:
#   DJF : [("2021", "12"), ("2022", "01"), ("2022", "02")]
#   MAM : [("2022", "03"), ("2022", "04"), ("2022", "05")]
#   JJA : [("2022", "06"), ("2022", "07"), ("2022", "08")]
#   SON : [("2022", "09"), ("2022", "10"), ("2022", "11")]
PERIODS = [
    ("2021", "12"),
    ("2022", "01"),
    ("2022", "02"),
]

# Template Path Input Data
# Variabel {year}, {month}, {month_name} akan disubstitusikan untuk setiap bulan di PERIODS.
HIMAWARI_INPUT_TEMPLATE = (
    "/data/mahasiswa/romadhon/perez1/himawari/{year}/"
    "himawari_{month_name}/nc/himawari_*.nc"
)
ERA5_SSRD_INPUT_TEMPLATE = (
    "/data/mahasiswa/romadhon/perez1/era5/ssrd/{year}/"
    "dataera5ssrd{month_name}.nc"
)
ERA5_TCWV_INPUT_TEMPLATE = (
    "/data/mahasiswa/romadhon/perez1/era5/tcwv/{year}/"
    "dataera5tcwv{month_name}.nc"
)
MASINGAR_INPUT_TEMPLATE = (
    "/data/mahasiswa/romadhon/perez1/masingar/aerosol/{year}/"
    "{month_name}/nc/dataaerosol_{year}{month}.nc"
)

# Direktori Output Dasar (Tersatu)
# Subfolder per bulan otomatis: <OUTPUT_BASE>/<TAHUN>/<BULAN_EN>/{nc,txt}/
OUTPUT_BASE = "/data/mahasiswa/romadhon/perez1/perez"

N_CPU = 45

COMMON_CONFIG = {
    "min_himawari_scene": 5,
    "max_sza": 80.0,
    "min_ssrd_wm2": 0.0,
    "na_thres": 1.0,  # Threshold fraksi NaN maksimum untuk adaptive masking xESMF (0.0 - 1.0)
    "scan_offsets_minutes": [-55, -45, -35, -25, -15, -5],
}


# ============================================================
# HELPER FUNCTIONS
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
        raise ValueError(
            f"Koordinat grid harus 1D dan minimal 2 titik. "
            f"Ditemukan dimensi {coord.ndim} dan panjang {len(coord)}."
        )

    if not np.all(np.diff(coord) > 0):
        raise ValueError(
            "Koordinat grid harus monoton naik untuk pembuatan bounds xESMF."
        )

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


def regrid_variable(regridder, data, na_thres=1.0):
    """Regrid satu DataArray latitude/longitude ke target grid dengan dukungan na_thres."""
    data = data.rename(
        {
            "latitude": "lat",
            "longitude": "lon",
        }
    )

    result = regridder(
        data,
        skipna=True,
        na_thres=na_thres,
    )

    return result.rename(
        {
            "lat": "latitude",
            "lon": "longitude",
        }
    )


def average_two_timepoints(ds, freq_offset=pd.Timedelta(hours=1)):
    """
    Merata-ratakan variabel dari dua titik waktu: (t - freq_offset) dan t.
    Hasilnya diberi label pada akhir jendela (t) agar selaras dengan data akumulasi 1 jam.
    """
    ds_shifted = ds.assign_coords(
        valid_time=ds.valid_time + freq_offset
    )

    common_t = ds.valid_time.to_index().intersection(ds_shifted.valid_time.to_index())
    if len(common_t) == 0:
        raise ValueError(
            "Tidak dapat merata-ratakan dua titik waktu: tidak ada irisan waktu berturutan."
        )

    ds_curr = ds.sel(valid_time=common_t)
    ds_prev = ds_shifted.sel(valid_time=common_t)

    ds_avg = (ds_curr + ds_prev) / 2.0
    ds_avg.attrs = ds.attrs
    return ds_avg


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
    print(f"  Tidak ditemukan  : {len(missing)}")

    if len(missing) > 0:
        print("  Contoh missing   :", list(missing[:5]))

    return common, missing


def compute_cosz_mean(time_end_utc, lat_arr, lon_arr, scan_offsets_minutes):
    """
    Menghitung cosz_mean sebagai rata-rata cos(SZA) atas enam slot scan Himawari-8
    dalam satu jendela 1 jam.

    Himawari-8 melakukan full-disk scan setiap 10 menit. Dalam jendela [t-60, t),
    terdapat enam slot dengan titik tengah masing-masing slot pada menit ke-55, 45,
    35, 25, 15, 5 sebelum label akhir jendela t (scan_offsets_minutes).

    Returns
    -------
    sza_midpoint : np.ndarray
        SZA (°) dihitung pada titik tengah jendela (offset -30 menit).
        Digunakan untuk kolom 'sza' dan perhitungan airmass.
    cosz_mean : np.ndarray
        Rata-rata cos(SZA) atas seluruh slot scan dalam satu jendela 1 jam.
    """
    n_slots = len(scan_offsets_minutes)
    n_rows = len(lat_arr)

    cosz_sum = np.zeros(n_rows, dtype=np.float64)

    for offset_min in scan_offsets_minutes:
        t_slot = time_end_utc + pd.Timedelta(minutes=offset_min)
        solar_slot = pvlib.solarposition.get_solarposition(
            time=t_slot,
            latitude=lat_arr,
            longitude=lon_arr,
        )
        cosz_slot = np.maximum(
            0.0,
            np.cos(np.radians(solar_slot["zenith"].to_numpy(dtype=float)))
        )
        cosz_sum += cosz_slot

    cosz_mean = cosz_sum / n_slots

    t_midpoint = time_end_utc + pd.Timedelta(minutes=-30)
    solar_midpoint = pvlib.solarposition.get_solarposition(
        time=t_midpoint,
        latitude=lat_arr,
        longitude=lon_arr,
    )
    sza_midpoint = solar_midpoint["zenith"].to_numpy(dtype=float)

    return sza_midpoint, cosz_mean


# ============================================================
# PEMROSESAN REGRIDDING PER BULAN
# ============================================================

def process_regrid_month(year, month, paths, cfg):
    ym = f"{year}{str(month).zfill(2)}"

    print("\n" + "=" * 70)
    print(f"PENYELARASAN SPASIAL-TEMPORAL & REGRIDDING: {year}-{str(month).zfill(2)} ({ym})")
    print("=" * 70)
    print(f"CPU tersedia   : {N_CPU}")
    print(f"Input Himawari : {paths['himawari']}")
    print(f"Input ERA5 SSRD: {paths['era5_ssrd']}")
    print(f"Input ERA5 TCWV: {paths['era5_tcwv']}")
    print(f"Input MASINGAR : {paths['masingar']}")
    print(f"Output NetCDF  : {paths['output_nc']}")
    print(f"Output TXT     : {paths['output_txt']}")
    print(f"xESMF na_thres : {cfg['na_thres']}")
    print(f"Slot scan cosz : {cfg['scan_offsets_minutes']} menit (6 slot tiap jam)")
    print("Catatan        : Tanpa filter jam pengamatan (semua jam diproses)")
    print("=" * 70)

    # 1. LOAD HIMAWARI
    print("\n" + "=" * 70)
    print("LOAD HIMAWARI-8 (KELUARAN PROCES6)")
    print("=" * 70)

    hima_files = glob.glob(paths["himawari"])
    if not hima_files:
        raise FileNotFoundError(f"File Himawari tidak ditemukan: {paths['himawari']}")

    hima = xr.open_mfdataset(
        paths["himawari"],
        engine="h5netcdf",
        combine="by_coords",
        chunks={"valid_time": 68},
    )

    hima = normalize_datetime(hima)

    if hima["latitude"].ndim != 1 or hima["longitude"].ndim != 1:
        raise ValueError(
            f"Koordinat Himawari harus 1-D regular lat/lon (seperti keluaran proces6). "
            f"Ditemukan dimensi latitude={hima['latitude'].dims}, longitude={hima['longitude'].dims}."
        )

    hima = hima.sortby("latitude")
    hima = hima.sortby("longitude")
    print(f"Jumlah file Himawari dibuka : {len(hima_files)}")
    print(f"Variabel Himawari          : {list(hima.data_vars)}")

    # 2. LOAD ERA5 SSRD (BULANAN)
    print("\n" + "=" * 70)
    print("LOAD ERA5 SSRD (BULANAN)")
    print("=" * 70)

    if not os.path.isfile(paths["era5_ssrd"]):
        raise FileNotFoundError(f"File ERA5 SSRD tidak ditemukan: {paths['era5_ssrd']}")

    era5ssrd = xr.open_dataset(paths["era5_ssrd"], engine="h5netcdf")
    era5ssrd = normalize_datetime(era5ssrd)
    era5ssrd = normalize_longitude(era5ssrd)
    era5ssrd = era5ssrd.sortby("latitude")
    print(f"Jumlah waktu awal ERA5 SSRD bulanan: {len(era5ssrd.valid_time)}")

    # 3. LOAD ERA5 TCWV (BULANAN) & RATA-RATA DUA TITIK WAKTU
    print("\n" + "=" * 70)
    print("LOAD ERA5 TCWV (BULANAN) & RATA-RATA DUA TITIK WAKTU")
    print("=" * 70)

    if not os.path.isfile(paths["era5_tcwv"]):
        raise FileNotFoundError(f"File ERA5 TCWV tidak ditemukan: {paths['era5_tcwv']}")

    era5tcwv = xr.open_dataset(paths["era5_tcwv"], engine="h5netcdf")
    era5tcwv = normalize_datetime(era5tcwv)
    era5tcwv = normalize_longitude(era5tcwv)
    era5tcwv = era5tcwv.sortby("latitude")
    print(f"Jumlah waktu awal ERA5 TCWV bulanan: {len(era5tcwv.valid_time)}")

    era5tcwv = average_two_timepoints(era5tcwv)
    print(f"Jumlah waktu ERA5 TCWV setelah rata-rata 2 titik waktu: {len(era5tcwv.valid_time)}")

    # 4. LOAD MASINGAR (BULANAN) & RATA-RATA DUA TITIK WAKTU
    print("\n" + "=" * 70)
    print("LOAD MASINGAR (BULANAN) & RATA-RATA DUA TITIK WAKTU")
    print("=" * 70)

    if not os.path.isfile(paths["masingar"]):
        raise FileNotFoundError(f"File MASINGAR tidak ditemukan: {paths['masingar']}")

    masingar_raw = xr.open_dataset(paths["masingar"], engine="h5netcdf")

    has_date = "date" in masingar_raw.variables
    has_time = "time" in masingar_raw.variables
    has_valid_time = (
        "valid_time" in masingar_raw.coords
        or "valid_time" in masingar_raw.variables
    )

    if has_date and has_time:
        print("Format Masingar : FLAT")
        required_columns = [
            "date", "time", "latitude", "longitude",
            "od500aer", "od550aer", "od870aer", "angstrom_turbidity",
        ]
        missing_columns = [col for col in required_columns if col not in masingar_raw.variables]
        if missing_columns:
            raise ValueError(f"Variabel Masingar tidak lengkap: {missing_columns}")

        masingar_df = masingar_raw[required_columns].to_dataframe().reset_index(drop=True)
        date_str = masingar_df["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
        time_num = pd.to_numeric(masingar_df["time"], errors="coerce")
        time_str = time_num.fillna(-1).astype(int).astype(str).str.zfill(2)
        masingar_df["valid_time"] = pd.to_datetime(date_str + time_str, format="%Y%m%d%H", errors="coerce")
        masingar_df = masingar_df.drop(columns=["date", "time"])
        masingar_df = masingar_df[masingar_df["valid_time"].notna()].copy()
        masingar_df["latitude"] = masingar_df["latitude"].astype(np.float64)
        masingar_df["longitude"] = masingar_df["longitude"].astype(np.float64)
        masingar_df = masingar_df.sort_values(["valid_time", "latitude", "longitude"])

        duplicated = masingar_df.duplicated(subset=["valid_time", "latitude", "longitude"])
        if duplicated.any():
            raise ValueError("Ditemukan duplikasi kombinasi valid_time-latitude-longitude pada Masingar.")

        masingar = masingar_df.set_index(["valid_time", "latitude", "longitude"]).to_xarray()
    elif has_valid_time:
        print("Format Masingar : SUDAH GRID")
        masingar = masingar_raw.copy()
    else:
        raise ValueError("Format Masingar tidak dikenali. Dataset harus memiliki date+time atau valid_time.")

    if "valid_time" not in masingar.coords:
        if "valid_time" in masingar.variables:
            masingar = masingar.set_coords("valid_time")
        else:
            raise ValueError("valid_time tidak ditemukan pada Masingar.")

    masingar = masingar.assign_coords(valid_time=pd.to_datetime(masingar.valid_time.values))
    masingar = masingar.sortby("valid_time")
    masingar = masingar.sortby("latitude")
    masingar = masingar.sortby("longitude")
    print(f"Jumlah waktu awal MASINGAR bulanan : {len(masingar.valid_time)}")

    masingar = average_two_timepoints(masingar)
    print(f"Jumlah waktu MASINGAR setelah rata-rata 2 titik waktu: {len(masingar.valid_time)}")

    # 5. TEMPORAL COVERAGE & HOURLY AVERAGING HIMAWARI
    print("\n" + "=" * 70)
    print("TEMPORAL AVERAGING HIMAWARI (LABEL AKHIR JENDELA & QC PER-PIKSEL)")
    print("=" * 70)

    hima_obs_variables = [var for var in ["radiance", "reflectance"] if var in hima.data_vars]
    if "reflectance" not in hima_obs_variables:
        raise ValueError("Variabel reflectance tidak ditemukan pada Himawari.")

    valid_refl_mask = np.isfinite(hima["reflectance"]) & (hima["reflectance"] > 0)
    valid_scan_count = (
        valid_refl_mask.astype(np.int32)
        .resample(valid_time="1h", label="right", closed="left")
        .sum(dim="valid_time")
    )

    hima_hourly = (
        hima[hima_obs_variables]
        .resample(valid_time="1h", label="right", closed="left")
        .mean(skipna=True)
    )

    hima_hourly = hima_hourly.where(valid_scan_count >= cfg["min_himawari_scene"])

    has_valid_pixels = (
        (valid_scan_count >= cfg["min_himawari_scene"])
        .any(dim=["latitude", "longitude"])
        .compute()
    )

    valid_hours = hima_hourly.valid_time.where(has_valid_pixels, drop=True).values
    hima_hourly = hima_hourly.sel(valid_time=hima_hourly.valid_time.isin(valid_hours))

    print(
        f"Jumlah jam Himawari setelah per-pixel coverage QC (>= {cfg['min_himawari_scene']} scan): "
        f"{len(hima_hourly.valid_time)}"
    )

    # 6. SINKRONISASI WAKTU DENGAN DATA BULANAN LAINNYA
    print("\n" + "=" * 70)
    print("SINKRONISASI WAKTU HIMAWARI vs DATA BULANAN")
    print("=" * 70)

    hima_times = pd.DatetimeIndex(pd.to_datetime(hima_hourly.valid_time.values))
    ssrd_times = pd.DatetimeIndex(pd.to_datetime(era5ssrd.valid_time.values))
    tcwv_times = pd.DatetimeIndex(pd.to_datetime(era5tcwv.valid_time.values))
    masingar_times = pd.DatetimeIndex(pd.to_datetime(masingar.valid_time.values))

    common_times = (
        hima_times
        .intersection(masingar_times)
        .intersection(ssrd_times)
        .intersection(tcwv_times)
    )

    print(f"Timestamp Himawari yang valid     : {len(hima_times)}")
    print(f"Timestamp MASINGAR bulanan        : {len(masingar_times)}")
    print(f"Timestamp ERA5 SSRD bulanan       : {len(ssrd_times)}")
    print(f"Timestamp ERA5 TCWV bulanan       : {len(tcwv_times)}")
    print(f"-> IRISAN WAKTU BERSAMA TERPILIH  : {len(common_times)}")

    if len(common_times) == 0:
        raise ValueError(
            f"Tidak ada irisan timestamp yang cocok untuk periode {ym}!\n"
            f"Contoh Himawari: {list(hima_times[:3])}\n"
            f"Contoh MASINGAR: {list(masingar_times[:3])}\n"
            f"Contoh SSRD    : {list(ssrd_times[:3])}\n"
            f"Pastikan tanggal dan jam yang diinput saling beririsan."
        )

    hima_hourly = hima_hourly.sel(valid_time=hima_hourly.valid_time.isin(common_times))
    masingar = masingar.sel(valid_time=masingar.valid_time.isin(common_times))
    era5ssrd = era5ssrd.sel(valid_time=era5ssrd.valid_time.isin(common_times))
    era5tcwv = era5tcwv.sel(valid_time=era5tcwv.valid_time.isin(common_times))

    print(f"✓ Semua dataset berhasil disinkronkan ke {len(common_times)} timestamp terpilih.")

    # 7. TARGET GRID & SOURCE GRIDS
    print("\n" + "=" * 70)
    print("PEMBUATAN GRID DAN REGRIDDER xESMF")
    print("=" * 70)

    target_lat = masingar.latitude.values.astype(np.float64)
    target_lon = masingar.longitude.values.astype(np.float64)
    target_grid = make_xesmf_grid(target_lat, target_lon)

    hima_lat = hima.latitude.values.astype(np.float64)
    hima_lon = hima.longitude.values.astype(np.float64)
    hima_source_grid = make_xesmf_grid(hima_lat, hima_lon)

    regridder_himawari = xe.Regridder(
        hima_source_grid,
        target_grid,
        method="conservative",
        periodic=False,
        reuse_weights=False,
    )
    print("✓ Regridder Himawari conservative berhasil dibuat")

    # 8. REGRIDDING HIMAWARI & ELEVATION
    print("\n" + "=" * 70)
    print("REGRIDDING HIMAWARI & ELEVATION")
    print("=" * 70)

    hima_regrid = xr.Dataset()
    for var in hima_obs_variables:
        print("Regridding:", var)
        hima_regrid[var] = regrid_variable(
            regridder_himawari,
            hima_hourly[var],
            na_thres=cfg["na_thres"],
        )

    if "elevation" in hima.variables or "elevation" in hima.coords:
        print("Regridding: elevation")
        elevation_source = hima["elevation"]
        if "valid_time" in elevation_source.dims:
            elevation_source = elevation_source.isel(valid_time=0, drop=True)
        hima_regrid["elevation"] = regrid_variable(
            regridder_himawari,
            elevation_source,
            na_thres=cfg["na_thres"],
        )
    else:
        print("Peringatan: Variabel elevation tidak ditemukan pada Himawari.")

    # 9. REGRIDDING ERA5 SSRD & TCWV
    print("\n" + "=" * 70)
    print("REGRIDDING ERA5 SSRD & TCWV")
    print("=" * 70)

    if "ssrd" not in era5ssrd.data_vars:
        raise ValueError("Variabel ssrd tidak ditemukan.")
    ssrd_source_grid = make_xesmf_grid(
        era5ssrd.latitude.values.astype(np.float64),
        era5ssrd.longitude.values.astype(np.float64),
    )
    regridder_ssrd = xe.Regridder(
        ssrd_source_grid, target_grid, method="conservative", periodic=False, reuse_weights=False
    )
    era5_ssrd_regrid = xr.Dataset(
        {"ssrd": regrid_variable(regridder_ssrd, era5ssrd["ssrd"], na_thres=cfg["na_thres"])}
    )
    print("✓ Regridding SSRD selesai")

    if "tcwv" not in era5tcwv.data_vars:
        raise ValueError("Variabel tcwv tidak ditemukan.")
    tcwv_source_grid = make_xesmf_grid(
        era5tcwv.latitude.values.astype(np.float64),
        era5tcwv.longitude.values.astype(np.float64),
    )
    regridder_tcwv = xe.Regridder(
        tcwv_source_grid, target_grid, method="conservative", periodic=False, reuse_weights=False
    )
    era5_tcwv_regrid = xr.Dataset(
        {"tcwv": regrid_variable(regridder_tcwv, era5tcwv["tcwv"], na_thres=cfg["na_thres"])}
    )
    print("✓ Regridding TCWV selesai")

    # 10. SET TARGET COORDINATES & AUDIT
    for ds in [hima_regrid, era5_ssrd_regrid, era5_tcwv_regrid]:
        ds.coords["latitude"] = masingar.latitude
        ds.coords["longitude"] = masingar.longitude

    exact_time_report(hima_regrid.valid_time.values, masingar.valid_time.values, "HIMAWARI vs MASINGAR")
    exact_time_report(era5_ssrd_regrid.valid_time.values, masingar.valid_time.values, "ERA5 SSRD vs MASINGAR")
    exact_time_report(era5_tcwv_regrid.valid_time.values, masingar.valid_time.values, "ERA5 TCWV vs MASINGAR")

    hima_temporal = hima_regrid.reindex(valid_time=masingar.valid_time)
    ssrd_temporal = era5_ssrd_regrid.reindex(valid_time=masingar.valid_time)
    tcwv_temporal = era5_tcwv_regrid.reindex(valid_time=masingar.valid_time)

    # 11. MERGE SEMUA DATA & DATAFRAME
    print("\n" + "=" * 70)
    print("MENGGABUNGKAN SEMUA DATA & KONVERSI DATAFRAME")
    print("=" * 70)

    combined = xr.merge(
        [masingar, hima_temporal, ssrd_temporal, tcwv_temporal],
        compat="override",
        join="exact",
    )

    df = combined.to_dataframe().reset_index()
    df = df.drop(columns=["number", "expver"], errors="ignore")
    print("Shape sebelum parameter target:", df.shape)

    # 12. SOLAR GEOMETRY & COSZ_MEAN
    print("\n" + "=" * 70)
    print("MENGHITUNG COSZ_MEAN (RATA-RATA 6 SLOT SCAN HIMAWARI-8 PER JAM)")
    print("=" * 70)

    time_end = pd.DatetimeIndex(pd.to_datetime(df["valid_time"]))
    if time_end.tz is None:
        time_end_utc = time_end.tz_localize("UTC")
    else:
        time_end_utc = time_end.tz_convert("UTC")

    lat_arr = df["latitude"].to_numpy(dtype=float)
    lon_arr = df["longitude"].to_numpy(dtype=float)

    sza_midpoint, cosz_mean = compute_cosz_mean(
        time_end_utc=time_end_utc,
        lat_arr=lat_arr,
        lon_arr=lon_arr,
        scan_offsets_minutes=cfg["scan_offsets_minutes"],
    )

    df["sza"] = sza_midpoint
    df["cosz_mean"] = cosz_mean

    # 13. AIR MASS, EPSILON, I0
    df["am"] = (
        pvlib.atmosphere.get_relative_airmass(
            df["sza"].to_numpy(dtype=float),
            model="kastenyoung1989",
        ).astype(float)
    )
    df.loc[~np.isfinite(df["am"]), "am"] = np.nan

    julianday = (time_end - pd.Timedelta(minutes=30)).dayofyear.to_numpy(dtype=float)
    df["epsilon"] = 1.0 + 0.033 * np.cos(2.0 * np.pi * julianday / 365.0)
    df["I0"] = 1361.0 * df["epsilon"]

    # 14. UNIT CONVERSIONS: SSRD (J/m² -> W/m²), TCWV (mm -> cm)
    if "ssrd" in df.columns:
        df["ssrd"] = df["ssrd"] / 3600.0

    if "tcwv" in df.columns:
        df["tcwv"] = df["tcwv"] / 10.0

    # 15. QC LENGKAP
    jumlah_awal = len(df)
    df = df[df["valid_time"].notna()].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if "radiance" in df.columns:
        df = df[np.isfinite(df["radiance"]) & (df["radiance"] > 0)].copy()

    df = df[np.isfinite(df["reflectance"]) & (df["reflectance"] > 0)].copy()
    df = df[np.isfinite(df["sza"]) & (df["sza"] < cfg["max_sza"])].copy()
    df = df[np.isfinite(df["ssrd"]) & (df["ssrd"] >= cfg["min_ssrd_wm2"])].copy()
    df = df[np.isfinite(df["tcwv"]) & (df["tcwv"] >= 0)].copy()
    df = df[np.isfinite(df["angstrom_turbidity"])].copy()

    df = df.drop(
        columns=[
            "radiance", "saa", "vza", "vaa", "od500aer", "od550aer",
            "od870aer", "time", "himawari_scene_count", "himawari_coverage",
            "angstrom_exponent",
        ],
        errors="ignore",
    )

    print("\n" + "=" * 70)
    print(f"FINAL QC ({ym})")
    print("=" * 70)
    print("Jumlah data awal :", f"{jumlah_awal:,}")
    print("Jumlah data akhir:", f"{len(df):,}")
    print("Data terbuang    :", f"{jumlah_awal - len(df):,}")
    if jumlah_awal > 0:
        print("Persentase lolos :", f"{len(df) / jumlah_awal * 100:.2f}%")

    # 16. RENAME & REORDER KOLOM
    rename_columns = {
        "angstrom_turbidity": "angstrom_turbidity(dimensionless)",
        "reflectance": "reflectance(dimensionless)",
        "sza": "sza(°)",
        "cosz_mean": "cosz_mean(dimensionless)",
        "am": "am(dimensionless)",
        "epsilon": "epsilon(dimensionless)",
        "I0": "I0(W/m²)",
        "elevation": "elevation(m)",
        "tcwv": "tcwv(cm)",
        "ssrd": "ssrd(W/m²)",
    }
    df = df.rename(columns=rename_columns)

    final_columns = [
        "valid_time", "latitude", "longitude",
        "angstrom_turbidity(dimensionless)", "reflectance(dimensionless)",
        "sza(°)", "cosz_mean(dimensionless)", "am(dimensionless)",
        "epsilon(dimensionless)", "I0(W/m²)", "elevation(m)",
        "tcwv(cm)", "ssrd(W/m²)",
    ]
    missing_final = [col for col in final_columns if col not in df.columns]
    if missing_final:
        raise ValueError(f"Kolom final tidak lengkap: {missing_final}")

    df = df[final_columns].copy()
    df = df.sort_values(["valid_time", "latitude", "longitude"])
    df["valid_time"] = pd.to_datetime(df["valid_time"])

    # 17. SIMPAN OUTPUT NETCDF & TXT
    print("\n" + "=" * 70)
    print("MENYIMPAN OUTPUT NETCDF DAN TXT")
    print("=" * 70)

    os.makedirs(os.path.dirname(paths["output_nc"]), exist_ok=True)
    os.makedirs(os.path.dirname(paths["output_txt"]), exist_ok=True)

    df_nc = df.copy().set_index(["valid_time", "latitude", "longitude"])
    ds_final = xr.Dataset.from_dataframe(df_nc)
    ds_final = ds_final.transpose("valid_time", "latitude", "longitude")
    ds_final = ds_final.rename({
        "angstrom_turbidity(dimensionless)": "angstrom_turbidity",
        "reflectance(dimensionless)": "reflectance",
        "sza(°)": "sza_deg",
        "cosz_mean(dimensionless)": "cosz_mean",
        "am(dimensionless)": "am",
        "epsilon(dimensionless)": "epsilon",
        "I0(W/m²)": "I0_Wm2",
        "elevation(m)": "elevation_m",
        "tcwv(cm)": "tcwv_cm",
        "ssrd(W/m²)": "ssrd_Wm2",
    })

    ds_final.to_netcdf(paths["output_nc"], engine="h5netcdf")
    print(f"✓ NetCDF berhasil disimpan: {paths['output_nc']}")

    df_txt = df.copy()
    df_txt["valid_time"] = df_txt["valid_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_txt.to_csv(paths["output_txt"], sep="\t", index=False)
    print(f"✓ TXT berhasil disimpan: {paths['output_txt']}")

    print(f"\n✓ SELESAI PEMROSESAN REGRIDDING BULAN {ym}")


# ============================================================
# MAIN PIPELINE 3 BULAN (1 MUSIM)
# ============================================================

def main():
    print("=" * 70)
    print("PENYELARASAN SPASIAL-TEMPORAL & REGRIDDING (REGRID10 V9 — 3 BULAN / 1 MUSIM)")
    print("=" * 70)
    print(f"Jumlah Periode Bulan : {len(PERIODS)} bulan")
    for idx, (yr, mo) in enumerate(PERIODS, 1):
        print(f"  Bulan ke-{idx}          : Tahun {yr}, Bulan {str(mo).zfill(2)}")
    print(f"Direktori Basis Out  : {OUTPUT_BASE}")
    print("Catatan              : Tanpa filter jam pengamatan (semua jam diproses)")
    print("=" * 70)

    for idx, (year_str, month_str) in enumerate(PERIODS, 1):
        year = str(year_str)
        month_2d = str(month_str).zfill(2)
        month_name = calendar.month_name[int(month_str)].lower()
        ym = f"{year}{month_2d}"

        # Subfolder otomatis berdasarkan tahun dan nama bulan bahasa Inggris: <OUTPUT_BASE>/<TAHUN>/<BULAN_EN>/{nc,txt}/
        output_nc_dir = os.path.join(OUTPUT_BASE, year, month_name, "nc")
        output_txt_dir = os.path.join(OUTPUT_BASE, year, month_name, "txt")
        os.makedirs(output_nc_dir, exist_ok=True)
        os.makedirs(output_txt_dir, exist_ok=True)

        output_nc = os.path.join(output_nc_dir, f"perez_{ym}.nc")
        output_txt = os.path.join(output_txt_dir, f"perez_{ym}.txt")

        hima_path = HIMAWARI_INPUT_TEMPLATE.format(year=year, month=month_2d, month_name=month_name)
        # Fallback fleksibel jika data Himawari berada di struktur folder format lama (month_MM)
        if not glob.glob(hima_path):
            legacy_hima_pattern = f"/data/mahasiswa/romadhon/perez1/himawari/{year}/month_{month_2d}/nc/himawari_*.nc"
            if glob.glob(legacy_hima_pattern):
                hima_path = legacy_hima_pattern

        ssrd_path = ERA5_SSRD_INPUT_TEMPLATE.format(year=year, month_name=month_name)
        tcwv_path = ERA5_TCWV_INPUT_TEMPLATE.format(year=year, month_name=month_name)

        masingar_path = MASINGAR_INPUT_TEMPLATE.format(year=year, month=month_2d, month_name=month_name)
        # Fallback fleksibel jika data aerosol MASINGAR berada di folder versi lama (1month/nc/)
        if not os.path.isfile(masingar_path):
            legacy_masingar = f"/data/mahasiswa/romadhon/perez1/masingar/aerosol/1month/nc/dataaerosol_{ym}.nc"
            if os.path.isfile(legacy_masingar):
                masingar_path = legacy_masingar

        paths = {
            "himawari": hima_path,
            "era5_ssrd": ssrd_path,
            "era5_tcwv": tcwv_path,
            "masingar": masingar_path,
            "output_nc": output_nc,
            "output_txt": output_txt,
        }

        print(f"\n>>> Memulai Regridding Periode [{idx}/{len(PERIODS)}]: {year}-{month_2d} ({month_name.upper()}) <<<")
        process_regrid_month(
            year=year,
            month=month_2d,
            paths=paths,
            cfg=COMMON_CONFIG,
        )

    print("\n" + "=" * 70)
    print("SELURUH PEMROSESAN 3 BULAN REGRIDDING SELESAI DENGAN SUKSES!")
    print("=" * 70)


if __name__ == "__main__":
    main()
