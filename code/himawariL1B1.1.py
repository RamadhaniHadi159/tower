import os
import datetime
import calendar
from concurrent.futures import ProcessPoolExecutor

from himawari_api import download_files

# ============================================================
# KONFIGURASI PERIODE DOWNLOAD (DAPAT DIUBAH DI SINI)
# ============================================================
# Tentukan tahun dan bulan awal serta tahun dan bulan akhir.
# Mendukung rentang lintas tahun (misalnya: Desember 2021 s/d Februari 2022)
START_YEAR = 2021
START_MONTH = 12

END_YEAR = 2022
END_MONTH = 2

# Validasi konfigurasi bulan dan tahun
if not (1 <= START_MONTH <= 12 and 1 <= END_MONTH <= 12):
    raise ValueError(
        f"START_MONTH ({START_MONTH}) dan END_MONTH ({END_MONTH}) harus berada pada rentang 1 s/d 12."
    )
if (START_YEAR, START_MONTH) > (END_YEAR, END_MONTH):
    raise ValueError(
        f"Periode awal ({START_YEAR}-{START_MONTH:02d}) tidak boleh lebih besar dari periode akhir ({END_YEAR}-{END_MONTH:02d})."
    )

# ============================================================
# CPU DARI SLURM
# ============================================================
n_cpu = int(
    os.environ.get(
        "SLURM_CPUS_PER_TASK",
        50
    )
)

print(f"CPU used : {n_cpu}")

# ============================================================
# PATH
# ============================================================
# base_dir dipertahankan sesuai struktur direktori yang ada.
# himawari_api menyusun output per tahun/bulan/hari/slot di bawah hierarki ini:
# <base_dir>/HIMAWARI-8/AHI-L1b-FLDK/<YEAR>/<MONTH>/<DAY>/<HHMM>/
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
def download_day(date_info):
    """
    Mengunduh data Himawari-8 L1B untuk satu hari penuh (00:00 - 23:50 UTC).
    
    Parameters
    ----------
    date_info : tuple (year, month, day) atau int (day)
        Informasi tanggal yang akan diunduh.
    """
    if isinstance(date_info, (tuple, list)):
        year, month, day = date_info
    else:
        # Fallback kompatibilitas jika dipanggil dengan parameter day saja
        year, month, day = START_YEAR, START_MONTH, date_info

    try:
        start_time = datetime.datetime(
            year, month, day, 0, 0
        )

        end_time = datetime.datetime(
            year, month, day, 23, 50
        )

        print(
            f"PID {os.getpid()} | "
            f"START DAY {year:04d}-{month:02d}-{day:02d}"
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

        n_files = len(l_fpaths) if l_fpaths is not None else 0

        print(
            f"PID {os.getpid()} | "
            f"FINISH DAY {year:04d}-{month:02d}-{day:02d} | "
            f"{n_files} files"
        )

        return n_files

    except Exception as e:
        print(
            f"PID {os.getpid()} | "
            f"ERROR DAY {year:04d}-{month:02d}-{day:02d} | {e}"
        )

        return 0

# ============================================================
# SEQUENTIAL MULTI-MONTH & PARALLEL DAILY DOWNLOAD
# ============================================================
if __name__ == "__main__":
    grand_total_files = 0
    monthly_summary = {}

    print("=" * 65)
    print("HIMAWARI L1B BATCH DOWNLOAD (MULTI-BULAN BERURUTAN)")
    print(f"Konfigurasi Periode : {START_YEAR}-{START_MONTH:02d} s/d {END_YEAR}-{END_MONTH:02d}")
    print(f"Alokasi CPU SLURM   : {n_cpu}")
    print(f"Base Directory      : {base_dir}")
    print("=" * 65)

    # Menghasilkan daftar pasangan (tahun, bulan) dari periode awal hingga akhir
    periods = []
    cur_year, cur_month = START_YEAR, START_MONTH
    while (cur_year < END_YEAR) or (cur_year == END_YEAR and cur_month <= END_MONTH):
        periods.append((cur_year, cur_month))
        if cur_month == 12:
            cur_month = 1
            cur_year += 1
        else:
            cur_month += 1

    # Loop berurutan per bulan
    for year, month in periods:
        # Tentukan jumlah hari tiap bulan secara otomatis (termasuk tahun kabisat)
        num_days = calendar.monthrange(year, month)[1]
        days = list(range(1, num_days + 1))
        tasks = [(year, month, d) for d in days]

        month_name = calendar.month_name[month]
        print(f"\n{'#' * 65}")
        print(f"MEMULAI PROSES BULAN: {year}-{month:02d} ({month_name.upper()}) | JUMLAH HARI: {num_days}")
        print(f"{'#' * 65}")

        # Paralelisasi hari per bulan menggunakan ProcessPoolExecutor berbasis SLURM
        max_workers = min(n_cpu, len(days))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                executor.map(
                    download_day,
                    tasks
                )
            )

        month_total = sum(results)
        grand_total_files += month_total
        monthly_summary[f"{year}-{month:02d}"] = month_total

        print(f"\n>>> SELESAI BULAN {year}-{month:02d} ({month_name}) | TOTAL FILE: {month_total} <<<")

    # Ringkasan akhir
    print("\n" + "=" * 65)
    print("SEMUA PROSES DOWNLOAD BERURUTAN SELESAI")
    print("Ringkasan File Terunduh per Bulan:")
    for month_key, total_count in monthly_summary.items():
        print(f"  - Bulan {month_key}: {total_count} files")
    print(f"TOTAL SELURUH FILE TERUNDUH: {grand_total_files}")
    print("=" * 65)
