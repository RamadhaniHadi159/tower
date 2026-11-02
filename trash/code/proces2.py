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
        # VALIDASI DASAR RADIANCE
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
        # SZA
        #
        # TIDAK MENGGUNAKAN FILTER SZA < 80°
        #
        # SZA tetap dihitung karena dibutuhkan untuk:
        # - cos(SZA)
        # - reflectance
        # - air mass
        # ====================================================
        cos_sza = np.cos(
            np.deg2rad(sza)
        )


        # ====================================================
        # CEGAH PEMBAGIAN DENGAN NILAI COS(SZA) ~ 0
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


        print(
            "Reflectance:",
            np.nanmin(reflectance),
            np.nanmean(reflectance),
            np.nanmax(reflectance)
        )

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

        print(
            "ERROR:",
            e
        )


        # ====================================================
        # HAPUS DAT JIKA ERROR
        # ====================================================
        for dat_file in dat_files:

            if os.path.exists(dat_file):

                os.remove(dat_file)


        return None


# ============================================================
# HANYA 1 HARI UNTUK TEST
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
    # FILTER AREA PENELITIAN
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
    # DIAGNOSTIK REFLECTANCE EKSTREM
    # ========================================================
    
    print()
    print("=" * 80)
    print("DIAGNOSTIK REFLECTANCE")
    print("=" * 80)
    
    # --------------------------------------------------------
    # Statistik dasar
    # --------------------------------------------------------
    print(
        "Jumlah data:",
        len(df_all)
    )
    
    print(
        "Reflectance min :",
        df_all["reflectance"].min()
    )
    
    print(
        "Reflectance mean:",
        df_all["reflectance"].mean()
    )
    
    print(
        "Reflectance max :",
        df_all["reflectance"].max()
    )
    
    
    # --------------------------------------------------------
    # Jumlah nilai berdasarkan threshold
    # --------------------------------------------------------
    thresholds = [
        1,
        2,
        5,
        10,
        50,
        100
    ]
    
    print()
    print("JUMLAH REFLECTANCE BERDASARKAN THRESHOLD")
    print("-" * 80)
    
    for threshold in thresholds:
    
        jumlah = (
            df_all["reflectance"] > threshold
        ).sum()
    
        persen = (
            jumlah /
            len(df_all) *
            100
        )
    
        print(
            f"Reflectance > {threshold:6} : "
            f"{jumlah:10d} "
            f"({persen:.6f} %)"
        )
    
    
    # --------------------------------------------------------
    # 20 nilai reflectance terbesar
    # --------------------------------------------------------
    print()
    print("=" * 80)
    print("20 REFLECTANCE TERBESAR")
    print("=" * 80)
    
    kolom_diagnostik = [
        "valid_time",
        "latitude",
        "longitude",
        "radiance",
        "reflectance",
        "sza",
        "saa",
        "vza",
        "vaa",
        "am"
    ]
    
    top20 = (
        df_all
        .sort_values(
            "reflectance",
            ascending=False
        )
        [kolom_diagnostik]
        .head(20)
    )
    
    print(
        top20.to_string(
            index=False
        )
    )
    
    
    # --------------------------------------------------------
    # Simpan diagnostik ke TXT
    # --------------------------------------------------------
    diagnostic_txt = os.path.join(
        output_txt_dir,
        f"diagnostic_reflectance_{year}{month}{day}.txt"
    )
    
    with open(
        diagnostic_txt,
        "w"
    ) as f:
    
        f.write(
            "=" * 80 + "\n"
        )
    
        f.write(
            "DIAGNOSTIK REFLECTANCE HIMAWARI-8\n"
        )
    
        f.write(
            f"TANGGAL : "
            f"{year}-{month}-{day}\n"
        )
    
        f.write(
            "=" * 80 + "\n\n"
        )
    
    
        # Statistik dasar
        f.write(
            "STATISTIK DASAR\n"
        )
    
        f.write(
            "-" * 80 + "\n"
        )
    
        f.write(
            f"Jumlah data      : "
            f"{len(df_all)}\n"
        )
    
        f.write(
            f"Reflectance min  : "
            f"{df_all['reflectance'].min()}\n"
        )
    
        f.write(
            f"Reflectance mean : "
            f"{df_all['reflectance'].mean()}\n"
        )
    
        f.write(
            f"Reflectance max  : "
            f"{df_all['reflectance'].max()}\n"
        )
    
    
        # Threshold
        f.write("\n")
        f.write(
            "JUMLAH REFLECTANCE BERDASARKAN THRESHOLD\n"
        )
    
        f.write(
            "-" * 80 + "\n"
        )
    
        for threshold in thresholds:
        
            jumlah = (
                df_all["reflectance"] > threshold
            ).sum()
    
            persen = (
                jumlah /
                len(df_all) *
                100
            )
    
            f.write(
                f"Reflectance > {threshold:6} : "
                f"{jumlah:10d} "
                f"({persen:.6f} %)\n"
            )
    
    
        # Top 20
        f.write("\n")
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
        "File diagnostik:",
        diagnostic_txt
    )
    print("=" * 80)
    
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
    # OUTPUT TXT
    # ========================================================
    output_txt = os.path.join(
        output_txt_dir,
        f"himawari_{year}{month}{day}_1day.txt"
    )


    with open(
        output_txt,
        "w"
    ) as f:

        f.write("=" * 80 + "\n")
        f.write(
            "HASIL PROCESSING HIMAWARI-8\n"
        )
        f.write(
            f"TANGGAL : "
            f"{year}-{month}-{day}\n"
        )
        f.write("=" * 80 + "\n\n")


        # ====================================================
        # INFORMASI DATA
        # ====================================================
        f.write(
            "INFORMASI DATASET\n"
        )

        f.write("-" * 80 + "\n")

        f.write(
            f"Jumlah baris       : "
            f"{len(df_all)}\n"
        )

        f.write(
            f"Jumlah waktu       : "
            f"{df_all['valid_time'].nunique()}\n"
        )

        f.write(
            f"Jumlah latitude    : "
            f"{df_all['latitude'].nunique()}\n"
        )

        f.write(
            f"Jumlah longitude   : "
            f"{df_all['longitude'].nunique()}\n"
        )

        f.write("\n")


        # ====================================================
        # STATISTIK
        # ====================================================
        f.write(
            "STATISTIK DATA\n"
        )

        f.write("-" * 80 + "\n")


        for kolom in [
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
        ]:

            if kolom in df_all.columns:

                data = df_all[kolom]

                f.write(
                    f"{kolom:15s} | "
                    f"min={data.min():.6f} | "
                    f"mean={data.mean():.6f} | "
                    f"max={data.max():.6f} | "
                    f"NaN={data.isna().sum()}\n"
                )


        # ====================================================
        # SAMPLE DATA
        # ====================================================
        f.write("\n")

        f.write(
            "=" * 80 + "\n"
        )

        f.write(
            "20 BARIS PERTAMA DATA\n"
        )

        f.write(
            "=" * 80 + "\n\n"
        )

        f.write(
            df_all.head(20).to_string(
                index=False
            )
        )


        f.write("\n\n")


        f.write(
            "=" * 80 + "\n"
        )

        f.write(
            "20 BARIS TERAKHIR DATA\n"
        )

        f.write(
            "=" * 80 + "\n\n"
        )

        f.write(
            df_all.tail(20).to_string(
                index=False
            )
        )

        f.write("\n")


    print(
        "SELESAI:",
        output_txt
    )


print("=" * 60)
print("TEST 1 HARI SELESAI")
print("=" * 60)