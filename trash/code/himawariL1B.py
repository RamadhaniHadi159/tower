#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Paraller download Himawari L1B 

import os
import datetime
from concurrent.futures import ProcessPoolExecutor

from himawari_api import download_files

# ============================================================
# CPU DARI SLURM
# ============================================================
n_cpu = int(
    os.environ.get(
        "SLURM_CPUS_PER_TASK",
        40
    )
)

print(f"CPU used : {n_cpu}")

# ============================================================
# PATH
# ============================================================
base_dir = (
    "/data/mahasiswa/romadhon/perez1/"
    "himawari/himawari_L1B"
)

protocol = "s3"
fs_args = {}

# ============================================================
# DATASET
# ============================================================
satellite = "himawari-8"
product_level = "L1B"
product = "Rad"

sector = "FLDK"

filter_parameters = {
    "channels": ["C03"],
    "scene_abbr": None
}

force_download = False

# ============================================================
# FUNCTION DOWNLOAD PER HARI
# ============================================================
def download_day(day):

    try:

        start_time = datetime.datetime(
            2022, 10, day, 0, 0
        )

        end_time = datetime.datetime(
            2022, 10, day, 23, 50
        )

        print(
            f"PID {os.getpid()} | "
            f"START DAY {day:02d}"
        )

        l_fpaths = download_files(
            base_dir=base_dir,
            protocol=protocol,
            fs_args=fs_args,
            satellite=satellite,
            product_level=product_level,
            product=product,
            sector=sector,
            start_time=start_time,
            end_time=end_time,
            filter_parameters=filter_parameters,

            # satu thread per process
            n_threads=1,

            force_download=force_download,
            check_data_integrity=True,
            progress_bar=False,
            verbose=False,
        )

        print(
            f"PID {os.getpid()} | "
            f"FINISH DAY {day:02d} | "
            f"{len(l_fpaths)} files"
        )

        return len(l_fpaths)

    except Exception as e:

        print(
            f"PID {os.getpid()} | "
            f"ERROR DAY {day:02d} | {e}"
        )

        return 0

# ============================================================
# LIST HARI BULAN 
# ============================================================
days = list(range(1, 32)) # perlu dirubah jika ganti sesuai bulan 
# (1.31, 2.28/29, 3.31, 4.30, 5.31, 6.30, 7.31, 8.31, 9.30, 10.31, 11.30, 12.31)

# ============================================================
# PARALLEL DOWNLOAD
# ============================================================
if __name__ == "__main__":

    with ProcessPoolExecutor(
        max_workers=min(
            n_cpu,
            len(days)
        )
    ) as executor:

        results = list(
            executor.map(
                download_day,
                days
            )
        )

    print("SEMUA DOWNLOAD SELESAI")
    print(f"TOTAL FILE : {sum(results)}")
