import calendar
import bz2
import glob
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pvlib
import rasterio
import xarray as xr

from pyresample.geometry import AreaDefinition
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from satpy import Scene


# ============================================================
# KONFIGURASI INPUT & OUTPUT (DAPAT DIUBAH DI SINI)
# ============================================================

# Jam Observasi Harian (UTC)
START_HOUR = 0              # Jam mulai UTC (0-23, integer)
END_HOUR = 24               # Jam akhir UTC (1-24, integer)

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

# Direktori Sumber Data Input
BASE_RAMSA = (
    "/data/mahasiswa/romadhon/perez1/himawari/"
    "himawari_L1B/HIMAWARI-8/AHI-L1b-FLDK"
)
DEM_FILE = (
    "/data/mahasiswa/romadhon/perez1/"
    "dem/elevasi/dem_jatim_cop30.tif"
)

# Direktori Output Dasar (Tersatu)
# Subfolder per bulan otomatis: <OUTPUT_BASE>/<TAHUN>/himawari_<BULAN_EN>/{nc,txt}/
OUTPUT_BASE = "/data/mahasiswa/romadhon/perez1/himawari"

# CPU Parallel
N_CPU = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "45")))


# ============================================================
# VALIDASI JAM & INISIALISASI
# ============================================================

if not (
    0 <= START_HOUR < 24
    and 0 < END_HOUR <= 24
    and START_HOUR < END_HOUR
):
    raise ValueError(
        f"START_HOUR ({START_HOUR}) dan END_HOUR ({END_HOUR}) harus memenuhi 0 <= START_HOUR < END_HOUR <= 24"
    )

print("=" * 70)
print("HIMAWARI L1B TO REGULAR GRID 0.02° (PROCES6 V8 — 3 BULAN / 1 MUSIM)")
print("=" * 70)
print(f"Rentang Jam UTC    : {START_HOUR:02d}:00 – {END_HOUR:02d}:00")
print(f"Jumlah Bulan       : {len(PERIODS)} bulan")
for idx, (yr, mo) in enumerate(PERIODS, 1):
    print(f"  Bulan ke-{idx}         : Tahun {yr}, Bulan {str(mo).zfill(2)}")
print(f"Direktori Input    : {BASE_RAMSA}")
print(f"File DEM           : {DEM_FILE}")
print(f"Direktori Basis Out: {OUTPUT_BASE}")
print(f"CPU digunakan      : {N_CPU}")
print("=" * 70)


# ============================================================
# DEFINISI TARGET GRID REGULER 0.02° JAWA TIMUR
# ============================================================

lon_min = 110.0
lon_max = 116.5
lat_min = -9.5
lat_max = -5.0
grid_res = 0.02

target_area_def = None
lons_1d = None
lats_1d = None
elevation_grid_2d = None


def init_002deg_geometry(lon_min, lat_min, lon_max, lat_max, res=0.02):
    width = int(round((lon_max - lon_min) / res)) + 1
    height = int(round((lat_max - lat_min) / res)) + 1

    area_extent = (
        lon_min - res / 2.0,
        lat_min - res / 2.0,
        lon_max + res / 2.0,
        lat_max + res / 2.0,
    )

    area_def = AreaDefinition(
        area_id="jatim_002deg",
        description="Jawa Timur regular lat/lon 0.02 deg",
        proj_id="latlon",
        projection={"proj": "latlong", "datum": "WGS84"},
        width=width,
        height=height,
        area_extent=area_extent,
    )

    lons = np.linspace(lon_min, lon_max, width, dtype=np.float64)
    lats = np.linspace(lat_min, lat_max, height, dtype=np.float64)

    return area_def, lons, lats


def compute_dem_mean_on_target_grid(dem_path, area_def):
    west, south, east, north = area_def.area_extent
    width = area_def.width
    height = area_def.height

    dst_transform = from_bounds(west, south, east, north, width, height)
    dst_crs = area_def.crs
    dem_mean = np.full((height, width), np.nan, dtype=np.float32)

    with rasterio.open(dem_path) as src:
        src_nodata = src.nodata if src.nodata is not None else -32767.0
        reproject(
            source=rasterio.band(src, 1),
            destination=dem_mean,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.average,
            src_nodata=src_nodata,
            dst_nodata=np.nan,
        )

    dem_mean[dem_mean <= -1000] = np.nan
    dem_mean[~np.isfinite(dem_mean)] = np.nan
    dem_mean = np.flipud(dem_mean)
    return dem_mean


# ============================================================
# PEMROSESAN SATU SCAN
# ============================================================

def proses_scene(bz2_files):
    t_start_scene = time.time()
    dat_files = []
    temp_extracted_files = []

    scene_label = bz2_files[0] if bz2_files else "<scene kosong>"

    try:
        print("=" * 60)
        print("PROCESSING SCENE:", scene_label)

        for bz2_file in bz2_files:
            if bz2_file.endswith(".bz2"):
                dat_file = bz2_file[:-4]
                dat_files.append(dat_file)

                if not os.path.exists(dat_file):
                    temp_dat = f"{dat_file}.tmp_{os.getpid()}"
                    with bz2.open(bz2_file, "rb") as f_in, open(temp_dat, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    os.replace(temp_dat, dat_file)
                    temp_extracted_files.append(dat_file)
            else:
                dat_files.append(bz2_file)

        scn = Scene(
            reader="ahi_hsd",
            filenames=dat_files
        )

        scn.load(
            ["B03"],
            calibration="radiance"
        )

        b03_attrs = scn["B03"].attrs
        time_params = b03_attrs.get("time_parameters", {})

        obs_start = (
            time_params.get("observation_start_time")
            or b03_attrs.get("start_time")
            or scn.start_time
        )
        obs_end = (
            time_params.get("observation_end_time")
            or b03_attrs.get("end_time")
            or scn.end_time
        )

        obs_start = pd.Timestamp(obs_start)
        obs_end = pd.Timestamp(obs_end)

        if (
            pd.isna(obs_start)
            or pd.isna(obs_end)
            or obs_end < obs_start
        ):
            raise ValueError(
                f"Rentang waktu observasi HSD tidak valid: "
                f"{obs_start} -- {obs_end}"
            )

        obs_time = obs_start + (obs_end - obs_start) / 2

        nom_start = (
            time_params.get("nominal_start_time")
            or b03_attrs.get("nominal_start_time")
            or scn.start_time
        )
        valid_time = pd.Timestamp(nom_start)
        if pd.isna(valid_time):
            valid_time = obs_start.floor("10min")

        print("Waktu nominal scan (valid_time)  :", valid_time)
        print("Waktu awal observasi HSD         :", obs_start)
        print("Waktu akhir observasi HSD        :", obs_end)
        print("Waktu observasi aktual (obs_time):", obs_time)

        scn_resampled = scn.resample(
            target_area_def,
            resampler="bucket_avg",
            fill_value=np.nan
        )
        b03_resampled = scn_resampled["B03"]

        radiance_2d = np.asarray(
            b03_resampled.values,
            dtype=np.float64
        )
        radiance_2d = np.flipud(radiance_2d)

        total_cells = radiance_2d.size
        nan_cells = int(np.isnan(radiance_2d).sum())
        valid_cells = total_cells - nan_cells
        pct_nan = (nan_cells / total_cells) * 100.0

        print(
            f"Statistik Sel Radiansi (bucket_avg): "
            f"Total={total_cells:,} | Valid={valid_cells:,} ({100 - pct_nan:.2f}%) | "
            f"NaN={nan_cells:,} ({pct_nan:.2f}%)"
        )

        lat_grid_2d, lon_grid_2d = np.meshgrid(lats_1d, lons_1d, indexing="ij")

        valid_mask = (
            np.isfinite(radiance_2d)
            & (lat_grid_2d >= lat_min) & (lat_grid_2d <= lat_max)
            & (lon_grid_2d >= lon_min) & (lon_grid_2d <= lon_max)
        )

        lat = lat_grid_2d[valid_mask]
        lon = lon_grid_2d[valid_mask]
        rad = radiance_2d[valid_mask]
        elevation = elevation_grid_2d[valid_mask]

        print("Jumlah titik radiansi valid lolos mask:", len(rad))
        if len(rad) == 0:
            print(f"Peringatan: Tidak ada radiansi valid pada scene: {scene_label}")
            return None

        times = pd.DatetimeIndex([obs_time] * len(lat))
        solar = pvlib.solarposition.get_solarposition(
            time=times,
            latitude=lat,
            longitude=lon
        )

        sza = solar["zenith"].to_numpy(dtype=float)
        saa = solar["azimuth"].to_numpy(dtype=float)
        cos_sza = np.cos(np.deg2rad(sza))

        julian_day = int(obs_time.dayofyear)
        d = 1.0 - 0.01672 * np.cos(np.deg2rad(0.9856 * (julian_day - 4)))
        d2 = d**2
        F0 = 1631.5726

        reflectance = np.full(rad.shape, np.nan, dtype=np.float64)
        daylight = np.isfinite(cos_sza) & (cos_sza > 0)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            reflectance[daylight] = (np.pi * rad[daylight] * d2) / (F0 * cos_sza[daylight])

        reflectance[~np.isfinite(reflectance)] = np.nan

        AM = np.asarray(
            pvlib.atmosphere.get_relative_airmass(sza, model="kastenyoung1989"),
            dtype=float
        )
        AM[~np.isfinite(AM)] = np.nan

        epsilon = 1.0 + 0.033 * np.cos(2.0 * np.pi * julian_day / 365.0)
        I0 = 1361.0 * epsilon

        valid_time_str = valid_time.strftime("%Y-%m-%d %H:%M:%S")
        obs_time_str = obs_time.strftime("%Y-%m-%d %H:%M:%S.%f")

        df = pd.DataFrame({
            "valid_time": valid_time_str,
            "latitude": lat,
            "longitude": lon,
            "obs_time": obs_time_str,
            "radiance": rad,
            "reflectance": reflectance,
            "sza": sza,
            "saa": saa,
            "am": AM,
            "epsilon": epsilon,
            "I0": I0,
            "elevation": elevation
        })

        rad_2d_out = np.where(valid_mask, radiance_2d, np.nan).astype(np.float32)
        ref_2d = np.full(radiance_2d.shape, np.nan, dtype=np.float32)
        sza_2d = np.full(radiance_2d.shape, np.nan, dtype=np.float32)
        saa_2d = np.full(radiance_2d.shape, np.nan, dtype=np.float32)
        am_2d = np.full(radiance_2d.shape, np.nan, dtype=np.float32)

        ref_2d[valid_mask] = reflectance
        sza_2d[valid_mask] = sza
        saa_2d[valid_mask] = saa
        am_2d[valid_mask] = AM

        ds_scene = xr.Dataset(
            data_vars={
                "radiance": (("valid_time", "latitude", "longitude"), rad_2d_out[np.newaxis, :, :]),
                "reflectance": (("valid_time", "latitude", "longitude"), ref_2d[np.newaxis, :, :]),
                "sza": (("valid_time", "latitude", "longitude"), sza_2d[np.newaxis, :, :]),
                "saa": (("valid_time", "latitude", "longitude"), saa_2d[np.newaxis, :, :]),
                "am": (("valid_time", "latitude", "longitude"), am_2d[np.newaxis, :, :]),
                "elevation": (("latitude", "longitude"), elevation_grid_2d.astype(np.float32)),
                "obs_time": (("valid_time",), [obs_time]),
                "epsilon": (("valid_time",), [float(epsilon)]),
                "I0": (("valid_time",), [float(I0)])
            },
            coords={
                "valid_time": [valid_time],
                "latitude": lats_1d.astype(np.float64),
                "longitude": lons_1d.astype(np.float64),
            }
        )

        t_elapsed_scene = time.time() - t_start_scene
        print(f"Durasi scene: {t_elapsed_scene:.2f} detik")

        return {"df": df, "ds": ds_scene, "valid_cells": valid_cells, "nan_cells": nan_cells}

    except Exception as exc:
        print("=" * 60, file=sys.stderr)
        print(f"[ERROR] Gagal memproses scene: {scene_label}", file=sys.stderr)
        print(f"Tipe Kesalahan : {type(exc).__name__}", file=sys.stderr)
        print(f"Pesan Kesalahan: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        return None

    finally:
        for temp_file in temp_extracted_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except OSError as exc:
                print(f"PERINGATAN gagal menghapus file sementara {temp_file}: {exc}", file=sys.stderr)


# ============================================================
# INISIALISASI GEOMETRI TARGET 0.02° DAN DEM (SEKALI DIAWAL)
# ============================================================

print("=" * 60)
print("Menginisialisasi geometri grid reguler 0.02°...")
target_area_def, lons_1d, lats_1d = init_002deg_geometry(
    lon_min, lat_min, lon_max, lat_max, res=grid_res
)
print(f"Dimensi grid reguler 0.02°: tinggi(lat)={len(lats_1d)}, lebar(lon)={len(lons_1d)}")

print("Menghitung elevasi DEM Copernicus rata-rata pada target grid...")
elevation_grid_2d = compute_dem_mean_on_target_grid(DEM_FILE, target_area_def)
print(f"DEM target grid: {np.isfinite(elevation_grid_2d).sum():,}/{elevation_grid_2d.size:,} sel valid")
print("=" * 60)


# ============================================================
# LOOP BESAR PEMROSESAN 3 BULAN DALAM 1 KODE
# ============================================================

total_months = len(PERIODS)

for month_idx, (year_str, month_str) in enumerate(PERIODS, 1):
    year = str(year_str)
    month = f"{int(month_str):02d}"
    month_name = calendar.month_name[int(month)].lower()

    # Struktur direktori output terpadu dengan penamaan bulan bahasa Inggris
    output_nc_dir = os.path.join(OUTPUT_BASE, year, f"himawari_{month_name}", "nc")
    output_txt_dir = os.path.join(OUTPUT_BASE, year, f"himawari_{month_name}", "txt")

    os.makedirs(output_nc_dir, exist_ok=True)
    os.makedirs(output_txt_dir, exist_ok=True)

    print("\n" + "#" * 70)
    print(f"MEMPROSES BULAN {month_idx}/{total_months}: TAHUN {year}, BULAN {month} ({month_name.upper()})")
    print(f"Folder Output NC : {output_nc_dir}")
    print(f"Folder Output TXT: {output_txt_dir}")
    print("#" * 70)

    num_days = calendar.monthrange(int(year), int(month))[1]
    days = [f"{d:02d}" for d in range(1, num_days + 1)]

    monthly_summary = []
    t_start_month = time.time()

    for day in days:
        t_start_day = time.time()
        current_date = f"{year}-{month}-{day}"
        print("\n" + "=" * 60)
        print(f"PROCESSING DAY: {current_date}; jam {START_HOUR:02d}–{END_HOUR:02d} UTC")
        print("=" * 60)

        try:
            all_scenes = []

            for hour in range(START_HOUR, END_HOUR):
                for minute in range(0, 60, 10):
                    folder = f"{hour:02d}{minute:02d}"
                    path = os.path.join(BASE_RAMSA, year, month, day, folder)

                    if not os.path.isdir(path):
                        continue

                    pattern_bz2 = os.path.join(path, "*B03*S0610*.DAT.bz2")
                    files = sorted(glob.glob(pattern_bz2))
                    if not files:
                        pattern_dat = os.path.join(path, "*B03*S0610*.DAT")
                        files = sorted(glob.glob(pattern_dat))

                    if files:
                        all_scenes.append(files)

            print("Jumlah scene segmen S0610 ditemukan:", len(all_scenes))

            if not all_scenes:
                t_elapsed_day = time.time() - t_start_day
                print(f"Tidak ada data untuk hari/jam ini pada tanggal {current_date}.")
                monthly_summary.append({
                    "date": current_date,
                    "status": "TIDAK ADA DATA",
                    "scenes_found": 0,
                    "scenes_success": 0,
                    "scenes_failed": 0,
                    "duration_sec": t_elapsed_day,
                    "output_nc": "-",
                    "output_txt": "-",
                    "note": "File/folder tidak ditemukan"
                })
                continue

            with ProcessPoolExecutor(max_workers=N_CPU) as executor:
                results = list(executor.map(proses_scene, all_scenes))

            failed_count = sum(res is None for res in results)
            success_count = len(all_scenes) - failed_count

            all_df = [res["df"] for res in results if res is not None and not res["df"].empty]
            all_ds = [res["ds"] for res in results if res is not None and res["ds"] is not None]

            total_valid_across_scenes = sum(res["valid_cells"] for res in results if res is not None)
            total_nan_across_scenes = sum(res["nan_cells"] for res in results if res is not None)
            total_cells_day = total_valid_across_scenes + total_nan_across_scenes

            print(f"Scene gagal/kosong: {failed_count}/{len(all_scenes)}")

            if total_cells_day > 0:
                pct_nan_day = (total_nan_across_scenes / total_cells_day) * 100.0
                print("\n" + "=" * 60)
                print("RINGKASAN KUALITAS SEL BUCKET_AVG HARIAN:")
                print(f"Total sel diuji      : {total_cells_day:,}")
                print(f"Total sel valid      : {total_valid_across_scenes:,} ({100 - pct_nan_day:.2f}%)")
                print(f"Total sel NaN        : {total_nan_across_scenes:,} ({pct_nan_day:.2f}%)")
                print("=" * 60)

            if not all_df:
                t_elapsed_day = time.time() - t_start_day
                print(f"Semua scene gagal atau tidak menghasilkan data untuk tanggal {current_date}.")
                monthly_summary.append({
                    "date": current_date,
                    "status": "GAGAL (KOSONG)",
                    "scenes_found": len(all_scenes),
                    "scenes_success": 0,
                    "scenes_failed": failed_count,
                    "duration_sec": t_elapsed_day,
                    "output_nc": "-",
                    "output_txt": "-",
                    "note": "Semua scene gagal / radiansi kosong"
                })
                continue

            df_all = pd.concat(all_df, ignore_index=True)

            duplicate_mask = df_all.duplicated(["valid_time", "latitude", "longitude"], keep=False)
            if duplicate_mask.any():
                print("PERINGATAN: ditemukan baris duplikat:", int(duplicate_mask.sum()))
                df_all = df_all.drop_duplicates(["valid_time", "latitude", "longitude"], keep="first")

            df_all = df_all.sort_values(["valid_time", "latitude", "longitude"])

            static_elevation = all_ds[0]["elevation"]
            all_ds_vars = [ds.drop_vars("elevation", errors="ignore") for ds in all_ds]
            ds = xr.concat(all_ds_vars, dim="valid_time")
            ds["elevation"] = static_elevation

            ds.attrs.update({
                "title": "Himawari-8 AHI B03 0.02deg regular grid radiance, reflectance, and solar geometry",
                "source": "Himawari-8 AHI L1B HSD (Segment S0610); Resampled to regular 0.02 deg lat-lon grid with bucket_avg",
                "time_reference": "valid_time: nominal scan schedule; obs_time: actual observation time from HSD metadata",
                "spatial_resolution": "0.02 degree regular lat-lon grid",
                "resampling_method": "Satpy bucket_avg (area-weighted sub-pixel average)",
                "elevation_source": "Copernicus DEM 30m averaged per 0.02deg grid cell",
                "note": "Reflectance is NaN when cos(SZA) <= 0; apply SZA < 80 degree QC downstream."
            })

            time_suffix = f"{START_HOUR:02d}-{END_HOUR:02d}"
            output_stem = f"himawari_{year}{month}{day}_{time_suffix}"
            output_nc = os.path.join(output_nc_dir, f"{output_stem}.nc")

            ds.to_netcdf(output_nc, engine="h5netcdf")
            print("SELESAI NETCDF:", output_nc)

            output_txt = os.path.join(output_txt_dir, f"{output_stem}.txt")
            df_all.to_csv(output_txt, sep="\t", index=False, float_format="%.3f", na_rep="NaN")
            print("SELESAI TXT:", output_txt)

            t_elapsed_day = time.time() - t_start_day
            print(f"Durasi total 1 hari : {t_elapsed_day:.2f} detik ({t_elapsed_day / 60:.2f} menit)")

            monthly_summary.append({
                "date": current_date,
                "status": "SUKSES",
                "scenes_found": len(all_scenes),
                "scenes_success": success_count,
                "scenes_failed": failed_count,
                "duration_sec": t_elapsed_day,
                "output_nc": output_nc,
                "output_txt": output_txt,
                "note": "OK"
            })

        except Exception as day_exc:
            t_elapsed_day = time.time() - t_start_day
            print(f"[ERROR] Terjadi kegagalan pemrosesan pada hari {current_date}: {day_exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            monthly_summary.append({
                "date": current_date,
                "status": "ERROR",
                "scenes_found": len(all_scenes) if 'all_scenes' in locals() else 0,
                "scenes_success": 0,
                "scenes_failed": len(all_scenes) if 'all_scenes' in locals() else 0,
                "duration_sec": t_elapsed_day,
                "output_nc": "-",
                "output_txt": "-",
                "note": f"Error: {day_exc}"
            })
            continue

    t_elapsed_month = time.time() - t_start_month
    total_days_in_month = len(days)
    success_days = sum(1 for rec in monthly_summary if rec["status"] == "SUKSES")
    no_data_days = sum(1 for rec in monthly_summary if rec["status"] == "TIDAK ADA DATA")
    failed_days = total_days_in_month - success_days - no_data_days

    print("\n" + "=" * 80)
    print(f"RINGKASAN PEMROSESAN BULAN {year}-{month} SELESAI")
    print(f"Total Hari: {total_days_in_month} | Sukses: {success_days} | Tanpa Data: {no_data_days} | Gagal: {failed_days}")
    print(f"Total Waktu: {t_elapsed_month / 60:.2f} menit")
    print("=" * 80)

print("\n" + "=" * 80)
print("SELURUH 3 BULAN DALAM PERIODE INI TELAH SELESAI DIPROSES.")
print("=" * 80)
