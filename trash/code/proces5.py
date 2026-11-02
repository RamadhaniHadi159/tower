#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import bz2
import glob
import shutil
import sys

import numpy as np
import pandas as pd
import pvlib
import rasterio
import xarray as xr

from satpy import Scene
from pyresample import create_area_def
from concurrent.futures import ProcessPoolExecutor


# ============================================================
# ARGUMENT JAM
# ============================================================

start_hour = int(sys.argv[1])
end_hour   = int(sys.argv[2])


# ============================================================
# CPU PARALLEL
# ============================================================

n_cpu = int(
    os.environ.get(
        "SLURM_CPUS_PER_TASK",
        30
    )
)

print(f"CPU digunakan: {n_cpu}")


# ============================================================
# PATH
# ============================================================

year  = "2022"
month = "09"

base_ramsa = (
    "/data/mahasiswa/romadhon/"
    "perez1/himawari/"
    "himawari_L1B/"
    "HIMAWARI-8/AHI-L1b-FLDK"
)

output_nc_dir = (
    "/data/mahasiswa/romadhon/"
    "perez1/himawari/2022/himawari_september/nc/"
)

output_txt_dir = (
    "/data/mahasiswa/romadhon/"
    "perez1/himawari/2022/himawari_september/txt/"
)

os.makedirs(
    output_nc_dir,
    exist_ok=True
)

os.makedirs(
    output_txt_dir,
    exist_ok=True
)

dem_file = (
    "/data/mahasiswa/romadhon/"
    "perez1/dem/elevasi/dem_jatim_cop30.tif"
)


# ============================================================
# LOAD DEM
# ============================================================

dem = rasterio.open(dem_file)

band_dem = dem.read(1)


# ============================================================
# AREA RESAMPLE
# ============================================================

area_def = create_area_def(

    area_id="jatim",

    projection={
        "proj": "latlong",
        "datum": "WGS84"
    },

    area_extent=[
        109.5,
        -9.0,
        116.5,
        -5.0
    ],

    resolution=0.02
)


# ============================================================
# FUNCTION PROCESS
# ============================================================

def proses_scene(bz2_files):

    dat_files = []

    try:

        print("=" * 60)
        print("PROCESSING SCENE")
        print(bz2_files[0])


        # ====================================================
        # EXTRACT SEMENTARA
        # ====================================================

        for bz2_file in bz2_files:

            dat_file = bz2_file.replace(
                ".bz2",
                ""
            )

            dat_files.append(dat_file)


            # ================================================
            # EXTRACT
            # ================================================

            if not os.path.exists(dat_file):

                print("EXTRACT:", bz2_file)

                with bz2.open(
                    bz2_file,
                    "rb"
                ) as f_in:

                    with open(
                        dat_file,
                        "wb"
                    ) as f_out:

                        shutil.copyfileobj(
                            f_in,
                            f_out
                        )


        # ====================================================
        # LOAD HIMAWARI
        # ====================================================

        scn = Scene(
            reader="ahi_hsd",
            filenames=dat_files
        )

        scn.load(
            ["B03"],
            calibration="radiance"
        )

        obs_time = scn.start_time


        # ====================================================
        # RESAMPLE
        # ====================================================

        scn = scn.resample(
            area_def
        )

        radiance = scn["B03"].values

        lons, lats = (
            scn["B03"]
            .area
            .get_lonlats()
        )

        lat = lats.flatten()
        lon = lons.flatten()
        rad = radiance.flatten()


        # ====================================================
        # FILTER VALID
        # ====================================================

        valid = np.isfinite(rad)

        lat = lat[valid]
        lon = lon[valid]
        rad = rad[valid]


        # ====================================================
        # SNAP GRID
        # ====================================================

        lat0 = -9.0
        lon0 = 109.5

        res = 0.02

        ilat = np.round(
            (lat - lat0) / res
        ).astype(int)

        ilon = np.round(
            (lon - lon0) / res
        ).astype(int)

        lat = lat0 + ilat * res
        lon = lon0 + ilon * res

        lat = lat.round(2)
        lon = lon.round(2)


        # ====================================================
        # SOLAR POSITION
        # ====================================================

        times = pd.DatetimeIndex(
            [obs_time] * len(lat)
        )

        solar = (
            pvlib.solarposition
            .get_solarposition(
                time=times,
                latitude=lat,
                longitude=lon
            )
        )

        sza = solar["zenith"].values
        saa = solar["azimuth"].values

        cos_sza = np.cos(
            np.deg2rad(sza)
        )

        print(
            "SZA:",
            np.nanmin(sza),
            np.nanmean(sza),
            np.nanmax(sza)
        )


        # ====================================================
        # SATELLITE ANGLE
        # ====================================================

        sat_lon = 140.7

        dlon = np.deg2rad(
            lon - sat_lon
        )

        lat_rad = np.deg2rad(lat)

        vza = np.rad2deg(
            np.arccos(
                np.cos(lat_rad) *
                np.cos(dlon)
            )
        )

        vaa = np.rad2deg(
            np.arctan2(
                np.sin(dlon),
                np.tan(lat_rad)
            )
        )


        # ====================================================
        # REFLECTANCE
        # ====================================================

        julianday = (
            pd.to_datetime(obs_time)
            .dayofyear
        )

        d = (
            1 -
            0.01672 *
            np.cos(
                np.deg2rad(
                    0.9856 *
                    (julianday - 4)
                )
            )
        )

        d2 = d**2

        F0 = 1631.5726

        with np.errstate(
            divide="ignore",
            invalid="ignore"
        ):

            reflectance = (
                np.pi *
                rad *
                d2
            ) / (
                F0 *
                cos_sza
            )

        print(
            "Reflectance:",
            np.nanmin(reflectance),
            np.nanmean(reflectance),
            np.nanmax(reflectance)
        )


        # ====================================================
        # AIR MASS
        # ====================================================

        AM = (
            pvlib.atmosphere
            .get_relative_airmass(
                sza,
                model="kastenyoung1989"
            )
            .astype(float)
        )

        AM[~np.isfinite(AM)] = np.nan

        print(
            "AM:",
            np.nanmin(AM),
            np.nanmean(AM),
            np.nanmax(AM)
        )


        # ====================================================
        # IRRADIANCE
        # ====================================================

        epsilon = (
            1 +
            0.033 *
            np.cos(
                2 *
                np.pi *
                julianday / 365
            )
        )

        I0 = 1361 * epsilon


        # ====================================================
        # DEM ELEVATION
        # ====================================================

        lat_dem = np.clip(
            lat,
            dem.bounds.bottom + 1e-6,
            dem.bounds.top - 1e-6
        )

        lon_dem = np.clip(
            lon,
            dem.bounds.left + 1e-6,
            dem.bounds.right - 1e-6
        )

        coords = list(
            zip(
                lon_dem,
                lat_dem
            )
        )

        elevation = np.array(
            [
                x[0]
                for x in dem.sample(coords)
            ],
            dtype=float
        )

        elevation[elevation <= -1000] = np.nan

        print(
            "NaN elevation:",
            np.isnan(elevation).sum(),
            "/",
            len(elevation)
        )


        # ====================================================
        # DATAFRAME
        # ====================================================

        df = pd.DataFrame({

            "valid_time": np.full(
                len(lat),
                pd.to_datetime(obs_time)
            ),

            "latitude": lat,

            "longitude": lon,

            "radiance": rad,

            "reflectance": reflectance,

            "sza": sza,
            "saa": saa,

            "vza": vza,
            "vaa": vaa,

            "am": AM,

            "epsilon": epsilon,

            "I0": I0,

            "elevation": elevation
        })


        # ====================================================
        # HAPUS FILE DAT
        # ====================================================

        for dat_file in dat_files:

            if os.path.exists(dat_file):

                os.remove(dat_file)

                print(
                    "DELETE DAT:",
                    dat_file
                )

        return df


    except Exception as e:

        print("ERROR:", e)


        # ====================================================
        # HAPUS DAT JIKA ERROR
        # ====================================================

        for dat_file in dat_files:

            if os.path.exists(dat_file):

                os.remove(dat_file)

        return None


# ============================================================
# LOOP DAY
# ============================================================

# ============================================================
# UJI COBA HARI 01
# ============================================================

days = [f"{d:02d}" for d in range(1, 32)]

# ============================================================
# JIKA SUDAH YAKIN DAN INGIN 1 BULAN:
#
# days = ["01"]
# ============================================================


for day in days:

    print("=" * 60)
    print(f"PROCESSING DAY: {day}")
    print("=" * 60)


    # ========================================================
    # KUMPULKAN SCENE
    # ========================================================

    all_scenes = []


    for hour in range(
        start_hour,
        end_hour
    ):

        for minute in range(
            0,
            60,
            10
        ):

            folder = f"{hour:02d}{minute:02d}"

            path = os.path.join(
                base_ramsa,
                year,
                month,
                day,
                folder
            )

            if not os.path.exists(path):
                continue

            files = sorted(
                glob.glob(
                    path +
                    "/*B03*.DAT.bz2"
                )
            )

            if len(files) > 0:

                all_scenes.append(files)


    print(
        "Jumlah scene:",
        len(all_scenes)
    )


    if len(all_scenes) == 0:

        print("Tidak ada data")

        continue


    # ========================================================
    # PARALLEL PROCESS
    # ========================================================

    with ProcessPoolExecutor(
        max_workers=n_cpu
    ) as executor:

        results = list(
            executor.map(
                proses_scene,
                all_scenes
            )
        )


    # ========================================================
    # GABUNG DATA
    # ========================================================

    all_df = [
        df for df in results
        if df is not None
    ]


    if len(all_df) == 0:

        print("Semua gagal")

        continue


    df_all = pd.concat(
        all_df,
        ignore_index=True
    )


    # ========================================================
    # FILTER AREA
    # ========================================================

    df_all = df_all[
        (
            (df_all["latitude"] >= -9.0)
            &
            (df_all["latitude"] <= -5.0)
        )
        &
        (
            (df_all["longitude"] >= 110.0)
            &
            (df_all["longitude"] <= 116.5)
        )
    ]


    # ========================================================
    # GROUPING
    # ========================================================

    df_all = df_all.groupby(
        [
            "valid_time",
            "latitude",
            "longitude"
        ],
        as_index=False
    ).mean()


    # ========================================================
    # SORT
    # ========================================================

    df_all = df_all.sort_values(
        [
            "valid_time",
            "latitude",
            "longitude"
        ]
    )


    # ========================================================
    # INDEX
    # ========================================================

    df_all["latitude"] = (
        df_all["latitude"]
        .astype(np.float32)
    )

    df_all["longitude"] = (
        df_all["longitude"]
        .astype(np.float32)
    )


    df_all = df_all.set_index([
        "valid_time",
        "latitude",
        "longitude"
    ])


    # ========================================================
    # XARRAY
    # ========================================================

    ds = xr.Dataset.from_dataframe(
        df_all
    )

    ds = ds.transpose(
        "valid_time",
        "latitude",
        "longitude"
    )


    # ========================================================
    # OUTPUT NETCDF
    # ========================================================

    output_nc = os.path.join(
        output_nc_dir,
        f"himawari_{year}{month}{day}.nc"
    )

    print(ds)

    ds.to_netcdf(
        output_nc,
        engine="h5netcdf"
    )

    print(
        "SELESAI:",
        output_nc
    )


    # ========================================================
    # OUTPUT TXT
    # ========================================================

    output_txt = os.path.join(
        output_txt_dir,
        f"himawari_{year}{month}{day}.txt"
    )

    df_txt = (
        ds
        .to_dataframe()
        .reset_index()
    )

    df_txt.to_csv(
        output_txt,
        sep="\t",
        index=False
    )

    print(
        "SELESAI TXT:",
        output_txt
    )


print("=" * 60)
print("SEMUA BULAN SELESAI")
print("=" * 60)
