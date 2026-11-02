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

from satpy import Scene
from pyresample import create_area_def
from concurrent.futures import ProcessPoolExecutor


# ============================================================
# ARGUMENT JAM
# ============================================================
start_hour = int(sys.argv[1])
end_hour   = int(sys.argv[2])


# ============================================================
# CPU
# ============================================================
n_cpu = int(
    os.environ.get(
        "SLURM_CPUS_PER_TASK",
        40
    )
)

print(f"CPU digunakan: {n_cpu}")


# ============================================================
# PATH
# ============================================================
year  = "2021"
month = "12"

base_ramsa = (
    "/data/mahasiswa/romadhon/"
    "perez/himawari/"
    "himawari_L1B/"
    "HIMAWARI-8/AHI-L1b-FLDK"
)

output_txt_dir = (
    "/data/mahasiswa/romadhon/"
    "perez1/himawari/"
    "himawari_2021/"
)

os.makedirs(
    output_txt_dir,
    exist_ok=True
)

dem_file = (
    "/data/mahasiswa/"
    "romadhon/perez/dem/"
    "dem_jatim_cop30.tif"
)


# ============================================================
# LOAD DEM
# ============================================================
dem = rasterio.open(dem_file)


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
# PROCESS SCENE
# ============================================================
def proses_scene(bz2_files):

    dat_files = []

    try:

        print("=" * 60)
        print("PROCESSING SCENE")
        print(bz2_files[0])


        # ====================================================
        # EXTRACT
        # ====================================================
        for bz2_file in bz2_files:

            dat_file = bz2_file.replace(
                ".bz2",
                ""
            )

            dat_files.append(dat_file)

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

        scn.load(["B03"])

        obs_time = scn.start_time


        # ====================================================
        # RESAMPLE
        # ====================================================
        scn = scn.resample(area_def)

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
        # VALID RADIANCE
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


        # ====================================================
        # COS SZA
        # ====================================================
        cos_sza = np.cos(
            np.deg2rad(sza)
        )


        # ====================================================
        # HANYA HINDARI PEMBAGIAN DENGAN NOL
        # ====================================================
        valid_cos = (
            np.isfinite(cos_sza)
            &
            (cos_sza > 1e-6)
        )

        lat = lat[valid_cos]
        lon = lon[valid_cos]
        rad = rad[valid_cos]

        sza = sza[valid_cos]
        saa = saa[valid_cos]
        cos_sza = cos_sza[valid_cos]


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

        reflectance = (
            np.pi *
            rad *
            d2
        ) / (
            F0 *
            cos_sza
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
        # DEM
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
        # DELETE DAT
        # ====================================================
        for dat_file in dat_files:

            if os.path.exists(dat_file):

                os.remove(dat_file)


        return df


    except Exception as e:

        print("ERROR:", e)

        for dat_file in dat_files:

            if os.path.exists(dat_file):

                os.remove(dat_file)

        return None


# ============================================================
# HANYA 1 HARI
# ============================================================
days = ["01"]


for day in days:

    print("=" * 60)
    print(
        f"PROCESSING DAY: "
        f"{year}-{month}-{day}"
    )
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

            folder = (
                f"{hour:02d}"
                f"{minute:02d}"
            )

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
    # PARALLEL
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
    # GABUNG
    # ========================================================
    all_df = [
        df
        for df in results
        if df is not None
    ]


    if len(all_df) == 0:

        print("Semua scene gagal")
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
    # DIAGNOSTIK SZA vs REFLECTANCE
    # ========================================================
    print()
    print("=" * 80)
    print("DIAGNOSTIK SZA vs REFLECTANCE")
    print("=" * 80)


    # --------------------------------------------------------
    # Rentang SZA
    # --------------------------------------------------------
    sza_bins = [
        0,
        60,
        70,
        75,
        80,
        85,
        88,
        89,
        89.5,
        89.9,
        89.99,
        90.0
    ]


    labels = [
        "<60",
        "60-70",
        "70-75",
        "75-80",
        "80-85",
        "85-88",
        "88-89",
        "89-89.5",
        "89.5-89.9",
        "89.9-89.99",
        "89.99-90"
    ]


    df_all["sza_group"] = pd.cut(
        df_all["sza"],
        bins=sza_bins,
        labels=labels,
        include_lowest=True
    )


    # --------------------------------------------------------
    # Statistik setiap kelompok SZA
    # --------------------------------------------------------
    print()

    print(
        f"{'SZA':15s}"
        f"{'N':12s}"
        f"{'Ref mean':15s}"
        f"{'Ref max':15s}"
        f"{'AM mean':15s}"
        f"{'AM max':15s}"
    )

    print("-" * 85)


    statistik = []


    for group in labels:

        subset = df_all[
            df_all["sza_group"] == group
        ]

        if len(subset) == 0:
            continue

        ref_mean = subset["reflectance"].mean()
        ref_max = subset["reflectance"].max()

        am_mean = subset["am"].mean()
        am_max = subset["am"].max()


        print(
            f"{group:15s}"
            f"{len(subset):12d}"
            f"{ref_mean:<15.6f}"
            f"{ref_max:<15.6f}"
            f"{am_mean:<15.6f}"
            f"{am_max:<15.6f}"
        )


        statistik.append({

            "SZA_group": group,

            "N": len(subset),

            "reflectance_mean": ref_mean,

            "reflectance_max": ref_max,

            "AM_mean": am_mean,

            "AM_max": am_max
        })


    statistik_df = pd.DataFrame(
        statistik
    )


    # ========================================================
    # JUMLAH REFLECTANCE > 1 PER SZA
    # ========================================================
    print()
    print("=" * 80)
    print("REFLECTANCE > 1 BERDASARKAN SZA")
    print("=" * 80)

    print()

    print(
        f"{'SZA':15s}"
        f"{'Total':12s}"
        f"{'>1':12s}"
        f"{'% >1':12s}"
    )

    print("-" * 55)


    for group in labels:

        subset = df_all[
            df_all["sza_group"] == group
        ]

        if len(subset) == 0:
            continue


        jumlah_gt1 = (
            subset["reflectance"] > 1
        ).sum()

        persen_gt1 = (
            jumlah_gt1 /
            len(subset) *
            100
        )


        print(
            f"{group:15s}"
            f"{len(subset):12d}"
            f"{jumlah_gt1:12d}"
            f"{persen_gt1:12.6f}"
        )


    # ========================================================
    # 20 DATA EKSTREM
    # ========================================================
    print()
    print("=" * 80)
    print("20 REFLECTANCE TERBESAR")
    print("=" * 80)

    top20 = (
        df_all
        .sort_values(
            "reflectance",
            ascending=False
        )
        [
            [
                "valid_time",
                "latitude",
                "longitude",
                "radiance",
                "reflectance",
                "sza",
                "am"
            ]
        ]
        .head(20)
    )

    print(
        top20.to_string(
            index=False
        )
    )


    # ========================================================
    # OUTPUT TXT
    # ========================================================
    output_txt = os.path.join(
        output_txt_dir,
        f"sza_reflectance_diagnostic_"
        f"{year}{month}{day}.txt"
    )


    with open(
        output_txt,
        "w"
    ) as f:

        f.write(
            "=" * 80 + "\n"
        )

        f.write(
            "DIAGNOSTIK SZA vs REFLECTANCE\n"
        )

        f.write(
            f"TANGGAL : "
            f"{year}-{month}-{day}\n"
        )

        f.write(
            "=" * 80 + "\n\n"
        )


        # ----------------------------------------------------
        # Statistik SZA
        # ----------------------------------------------------
        f.write(
            "STATISTIK BERDASARKAN RENTANG SZA\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        f.write(
            statistik_df.to_string(
                index=False
            )
        )


        # ----------------------------------------------------
        # Reflectance > 1
        # ----------------------------------------------------
        f.write("\n\n")

        f.write(
            "=" * 80 + "\n"
        )

        f.write(
            "REFLECTANCE > 1 BERDASARKAN SZA\n"
        )

        f.write(
            "=" * 80 + "\n\n"
        )


        for group in labels:

            subset = df_all[
                df_all["sza_group"] == group
            ]

            if len(subset) == 0:
                continue

            jumlah_gt1 = (
                subset["reflectance"] > 1
            ).sum()

            persen_gt1 = (
                jumlah_gt1 /
                len(subset) *
                100
            )

            f.write(
                f"{group:15s}"
                f" Total={len(subset):10d}"
                f" >1={jumlah_gt1:10d}"
                f" %={persen_gt1:.6f}\n"
            )


        # ----------------------------------------------------
        # Top 20
        # ----------------------------------------------------
        f.write("\n\n")

        f.write(
            "=" * 80 + "\n"
        )

        f.write(
            "20 REFLECTANCE TERBESAR\n"
        )

        f.write(
            "=" * 80 + "\n\n"
        )

        f.write(
            top20.to_string(
                index=False
            )
        )

        f.write("\n")


    print()
    print(
        "FILE DIAGNOSTIK:",
        output_txt
    )


print("=" * 80)
print("DIAGNOSTIK SELESAI")
print("=" * 80)