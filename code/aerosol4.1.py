import calendar
import glob
import os
import re

import numpy as np
import pandas as pd
import xarray as xr


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

# Template Path Input MASINGAR
# Variabel {year} dan {month} disubstitusikan untuk setiap bulan di PERIODS.
INPUT_DIR_TEMPLATE = (
    "/data/mahasiswa/romadhon/perez1/masingar/{year}/{year}{month}/"
)

# Direktori Output Dasar (Tersatu)
# Subfolder per bulan otomatis: <OUTPUT_BASE>/<TAHUN>/<BULAN_EN>/{nc,txt}/
OUTPUT_BASE = "/data/mahasiswa/romadhon/perez1/masingar/aerosol/"

# Pengaturan Umum Domain & Fisik Aerosol
COMMON_CONFIG = {
    # Batas domain Jawa Timur
    "lon_min": 110.0,
    "lon_max": 116.5,
    "lat_min": -9.5,
    "lat_max": -5.0,
    # Ambang batas fisik AOD
    "min_aod_threshold": 1e-4,  # Nilai minimum AOD positif yang valid
    "alpha_min": -0.5,          # Batas bawah fisik Angstrom exponent
    "alpha_max": 4.0,           # Batas atas fisik Angstrom exponent
    # Pengaturan filter & kualitas data
    "filter_anl_only": True,    # Hanya memproses file analisis ('_ANL' in filename)
    "apply_alpha_clipping": True,  # Batasi alpha ke [alpha_min, alpha_max] jika ditemukan di luar rentang fisik
}


# ============================================================
# FUNGSI PEMROSESAN PER BULAN
# ============================================================

def process_aerosol_month(year, month, input_dir, output_nc, output_txt, cfg):
    """Memproses data aerosol MASINGAR untuk satu bulan tertentu."""
    ym = f"{year}{str(month).zfill(2)}"

    print("\n" + "=" * 70)
    print(f"PENGOLAHAN DATA AEROSOL MASINGAR: {year}-{str(month).zfill(2)} ({ym})")
    print("=" * 70)
    print(f"Direktori input      : {input_dir}")
    print(f"Target NetCDF        : {output_nc}")
    print(f"Target TXT           : {output_txt}")
    print(f"Domain Lat           : [{cfg['lat_min']}, {cfg['lat_max']}]")
    print(f"Domain Lon           : [{cfg['lon_min']}, {cfg['lon_max']}]")
    print(f"Filter file analisis : {cfg['filter_anl_only']} (Pemeriksaan: '_ANL' in filename)")
    print("=" * 70)

    # 1. PENCARIAN & PENYARINGAN FILE NETCDF (HANYA _ANL)
    all_nc_files = sorted(
        glob.glob(
            os.path.join(
                input_dir,
                "**",
                "*.nc"
            ),
            recursive=True
        )
    )

    print(f"Jumlah seluruh file .nc ditemukan : {len(all_nc_files)}")

    if len(all_nc_files) == 0:
        raise FileNotFoundError(
            f"Tidak ditemukan file .nc di dalam direktori:\n{input_dir}"
        )

    # Penyaringan ketat: hanya memproses file analisis dengan substring '_ANL'
    if cfg["filter_anl_only"]:
        files = [f for f in all_nc_files if "_ANL" in os.path.basename(f)]
        non_anl_count = len(all_nc_files) - len(files)
        print(f"Jumlah file analisis (_ANL)       : {len(files)}")
        print(f"Jumlah file non-ANL diabaikan     : {non_anl_count}")
    else:
        files = all_nc_files
        print(f"Filter ANL dinonaktifkan. Jumlah file yang diproses: {len(files)}")

    if len(files) == 0:
        raise FileNotFoundError(
            f"Tidak ditemukan file analisis (memuat '_ANL') di dalam direktori:\n{input_dir}\n"
            f"(Dari {len(all_nc_files)} file .nc, tidak ada yang memuat substring '_ANL' pada nama filenya)"
        )

    # 2. PROSES SETIAP FILE SECARA AMAN DENGAN DETEKSI DUPLIKAT
    hasil = []
    file_dilewati = []
    jumlah_file_berhasil = 0

    seen_timestamps = {}  # valid_time -> filename

    total_alpha_points = 0
    total_alpha_below = 0
    total_alpha_above = 0

    required_vars = ["od500aer", "od550aer", "od870aer"]

    for i, input_file in enumerate(files, start=1):
        filename = os.path.basename(input_file)

        match = re.search(
            r"H08_(\d{8})_(\d{4})_",
            filename
        )

        if match is None:
            print(f"PERINGATAN [{i}]: Format nama file tidak dikenali: {filename}")
            file_dilewati.append((filename, "Format nama file tidak cocok"))
            continue

        date_str = match.group(1)
        time_str = match.group(2)

        try:
            valid_time = pd.to_datetime(
                date_str + time_str,
                format="%Y%m%d%H%M"
            )
        except Exception as e:
            print(f"PERINGATAN [{i}]: Gagal parsing waktu {date_str}{time_str}: {e}")
            file_dilewati.append((filename, f"Error parsing waktu: {e}"))
            continue

        # Validasi duplikasi waktu: Hentikan jika ditemukan waktu duplikat
        if valid_time in seen_timestamps:
            prev_file = seen_timestamps[valid_time]
            raise ValueError(
                f"WAKTU DUPLIKAT TERDETEKSI!\n"
                f"Timestamp {valid_time} sudah ada dari file:\n  {prev_file}\n"
                f"dan ditemukan kembali pada file:\n  {filename}\n"
                "Proses dihentikan dengan galat sesuai kebijakan integritas data temporal."
            )
        seen_timestamps[valid_time] = filename

        # Buka dataset NetCDF & Validasi variabel
        try:
            with xr.open_dataset(input_file, engine="netcdf4") as ds:
                missing_vars = [v for v in required_vars if v not in ds.variables]
                if missing_vars:
                    print(f"PERINGATAN [{i}]: File {filename} tidak memiliki variabel {missing_vars}")
                    file_dilewati.append((filename, f"Variabel hilang: {missing_vars}"))
                    continue

                if "lat" not in ds.coords and "lat" not in ds.variables:
                    print(f"PERINGATAN [{i}]: Koordinat lat tidak ditemukan pada {filename}")
                    file_dilewati.append((filename, "Koordinat lat hilang"))
                    continue

                if "lon" not in ds.coords and "lon" not in ds.variables:
                    print(f"PERINGATAN [{i}]: Koordinat lon tidak ditemukan pada {filename}")
                    file_dilewati.append((filename, "Koordinat lon hilang"))
                    continue

                # Pemotongan Spasial (Subset Jawa Timur)
                if ds.lat.values[0] < ds.lat.values[-1]:
                    ds_jatim = ds.sel(
                        lat=slice(cfg["lat_min"], cfg["lat_max"]),
                        lon=slice(cfg["lon_min"], cfg["lon_max"])
                    )
                else:
                    ds_jatim = ds.sel(
                        lat=slice(cfg["lat_max"], cfg["lat_min"]),
                        lon=slice(cfg["lon_min"], cfg["lon_max"])
                    )

                aod500 = ds_jatim["od500aer"].load()
                aod550 = ds_jatim["od550aer"].load()
                aod870 = ds_jatim["od870aer"].load()

                # Penanganan AOD nol atau negatif & nilai tak-hingga
                valid_mask = (
                    np.isfinite(aod500) & (aod500 >= cfg["min_aod_threshold"]) &
                    np.isfinite(aod550) & (aod550 >= cfg["min_aod_threshold"]) &
                    np.isfinite(aod870) & (aod870 >= cfg["min_aod_threshold"])
                )

                aod500_clean = xr.where(valid_mask, aod500, np.nan)
                aod550_clean = xr.where(valid_mask, aod550, np.nan)
                aod870_clean = xr.where(valid_mask, aod870, np.nan)

                # Perhitungan Ångström Exponent Mentah (alpha_raw)
                # Formula: alpha = ln(AOD500 / AOD870) / ln(0.870 / 0.500)
                ratio_500_870 = aod500_clean / aod870_clean
                log_wavelength_ratio = np.log(0.870 / 0.500)

                valid_ratio_mask = valid_mask & (ratio_500_870 > 0) & np.isfinite(ratio_500_870)
                alpha_raw = xr.where(
                    valid_ratio_mask,
                    np.log(ratio_500_870) / log_wavelength_ratio,
                    np.nan
                )

                # Evaluasi Rentang Fisik alpha [-0.5, 4.0] Sebelum Hitung beta
                finite_alpha_mask = np.isfinite(alpha_raw.values)
                alpha_vals = alpha_raw.values[finite_alpha_mask]

                n_file_alpha = int(finite_alpha_mask.sum())
                n_file_below = int(np.sum(alpha_vals < cfg["alpha_min"]))
                n_file_above = int(np.sum(alpha_vals > cfg["alpha_max"]))
                n_file_out = n_file_below + n_file_above

                total_alpha_points += n_file_alpha
                total_alpha_below += n_file_below
                total_alpha_above += n_file_above

                if cfg["apply_alpha_clipping"] and (n_file_out > 0):
                    alpha = xr.where(
                        (alpha_raw >= cfg["alpha_min"]) & (alpha_raw <= cfg["alpha_max"]),
                        alpha_raw,
                        np.nan
                    )
                else:
                    alpha = alpha_raw.copy()

                # Perhitungan Ångström Turbidity (beta)
                # Formula: beta = AOD550 * (0.550 ** alpha)
                beta = xr.where(
                    valid_mask & np.isfinite(alpha),
                    aod550_clean * (0.550 ** alpha),
                    np.nan
                )

                ds_aerosol = xr.Dataset(
                    {
                        "od500aer": aod500_clean,
                        "od550aer": aod550_clean,
                        "od870aer": aod870_clean,
                        "angstrom_exponent_raw": alpha_raw,
                        "angstrom_exponent": alpha,
                        "angstrom_turbidity": beta,
                    }
                )

                ds_aerosol = ds_aerosol.rename(
                    {
                        "lat": "latitude",
                        "lon": "longitude",
                    }
                )

                ds_aerosol = ds_aerosol.expand_dims(
                    valid_time=[valid_time]
                )

                ds_aerosol = ds_aerosol.load()
                hasil.append(ds_aerosol)
                jumlah_file_berhasil += 1

        except Exception as e:
            if "WAKTU DUPLIKAT TERDETEKSI" in str(e):
                raise
            print(f"ERROR [{i}]: Gagal memproses file {filename}: {e}")
            file_dilewati.append((filename, str(e)))
            continue

        if i % 100 == 0 or i == len(files):
            print(f"Progress: {i}/{len(files)} file diproses (Berhasil: {jumlah_file_berhasil})")

    # 3. PENGGABUNGAN DATASET (JOIN='EXACT' TANPA FALLBACK)
    print("\n" + "=" * 70)
    print("PENGGABUNGAN DATA DAN VALIDASI GRID SPASIAL")
    print("=" * 70)

    if len(hasil) == 0:
        raise RuntimeError(
            f"Tidak ada data yang berhasil diproses untuk {ym}. Periksa format file dan variabel input."
        )

    print(f"Total file berhasil diproses : {jumlah_file_berhasil}")
    print(f"Total file dilewati          : {len(file_dilewati)}")

    try:
        ds_final = xr.concat(
            hasil,
            dim="valid_time",
            join="exact"
        )
        print("✓ Validasi grid berhasil: seluruh file memiliki koordinat spasial yang identik (join='exact').")
    except ValueError as e:
        raise ValueError(
            f"Gagal menggabungkan dataset {ym}: Grid spasial antar file NetCDF tidak cocok secara eksak!\n"
            f"Detail galat xarray: {e}\n"
            "Sesuai kebijakan integritas spasial, fallback join dihilangkan dan proses dihentikan."
        ) from e

    # 4. PENGURUTAN KOORDINAT & VALIDASI AKHIR WAKTU DUPLIKAT
    ds_final = ds_final.sortby("valid_time")
    ds_final = ds_final.sortby("latitude")
    ds_final = ds_final.sortby("longitude")

    time_values = pd.DatetimeIndex(ds_final.valid_time.values)
    duplicated_mask = time_values.duplicated(keep=False)
    jumlah_duplikat = int(time_values.duplicated(keep="first").sum())

    if jumlah_duplikat > 0:
        duplicate_times = time_values[duplicated_mask].unique()
        sample_dups = [str(t) for t in duplicate_times[:10]]
        raise ValueError(
            f"Ditemukan {jumlah_duplikat} timestamp duplikat pada dataset gabungan {ym}!\n"
            f"Contoh waktu duplikat: {sample_dups}\n"
            "Proses dihentikan dengan galat karena duplikasi waktu tidak ditoleransi."
        )

    print(f"✓ Validasi temporal berhasil: seluruh {len(ds_final.valid_time)} waktu bersifat unik (tanpa duplikat).")

    # 5. PEMERIKSAAN JAM YANG HILANG (TEMPORAL COMPLETENESS)
    time_index = pd.DatetimeIndex(ds_final.valid_time.values)
    t_min = time_index.min()
    t_max = time_index.max()

    full_hourly_range = pd.date_range(start=t_min, end=t_max, freq="1h")
    missing_hours = full_hourly_range.difference(time_index)

    print("\nRingkasan Kelengkapan Waktu:")
    print(f" - Waktu awal tersedia  : {t_min}")
    print(f" - Waktu akhir tersedia : {t_max}")
    print(f" - Jumlah jam diharapkan: {len(full_hourly_range)}")
    print(f" - Jumlah jam tersedia  : {len(time_index)}")
    print(f" - Jumlah jam hilang    : {len(missing_hours)} ({len(missing_hours)/len(full_hourly_range)*100:.2f}%)")

    # 6. EVALUASI RENTANG FISIK ÅNGSTRÖM EXPONENT (ALPHA)
    print("\n" + "=" * 70)
    print("EVALUASI RENTANG FISIK ÅNGSTRÖM EXPONENT (ALPHA)")
    print("=" * 70)
    total_out = total_alpha_below + total_alpha_above
    pct_out = (total_out / total_alpha_points * 100.0) if total_alpha_points > 0 else 0.0
    pct_below = (total_alpha_below / total_alpha_points * 100.0) if total_alpha_points > 0 else 0.0
    pct_above = (total_alpha_above / total_alpha_points * 100.0) if total_alpha_points > 0 else 0.0

    print(f"Total titik alpha valid terhitung  : {total_alpha_points:,}")
    print(f"Titik alpha < {cfg['alpha_min']} (non-fisik bawah) : {total_alpha_below:,} ({pct_below:.3f}%)")
    print(f"Titik alpha > {cfg['alpha_max']} (non-fisik atas)  : {total_alpha_above:,} ({pct_above:.3f}%)")
    print(f"Total titik di luar rentang        : {total_out:,} ({pct_out:.3f}%)")

    # 7. PENYUSUNAN METADATA & ATRIBUT STANDAR
    ds_final["od500aer"].attrs = {
        "long_name": "Aerosol optical depth at 500 nm",
        "units": "1",
        "valid_min": float(cfg["min_aod_threshold"]),
    }
    ds_final["od550aer"].attrs = {
        "long_name": "Aerosol optical depth at 550 nm",
        "units": "1",
        "valid_min": float(cfg["min_aod_threshold"]),
    }
    ds_final["od870aer"].attrs = {
        "long_name": "Aerosol optical depth at 870 nm",
        "units": "1",
        "valid_min": float(cfg["min_aod_threshold"]),
    }
    ds_final["angstrom_exponent_raw"].attrs = {
        "long_name": "Raw Angstrom exponent (500-870 nm) without physical clipping",
        "units": "1",
        "formula": "ln(AOD500/AOD870) / ln(0.870/0.500)",
    }
    ds_final["angstrom_exponent"].attrs = {
        "long_name": "Angstrom exponent (500-870 nm) constrained to physically realistic bounds",
        "units": "1",
        "formula": "ln(AOD500/AOD870) / ln(0.870/0.500)",
        "valid_range": [float(cfg["alpha_min"]), float(cfg["alpha_max"])],
    }
    ds_final["angstrom_turbidity"].attrs = {
        "long_name": "Angstrom turbidity coefficient (beta)",
        "units": "1",
        "formula": "AOD550 * (0.550 ** alpha)",
    }
    ds_final.attrs = {
        "title": "MASINGAR Aerosol Optical Depth and Angstrom Parameters (Quality Controlled)",
        "source": "MASINGAR (Model of Aerosol Species IN the Global AtmospheRe, JMA/MRI)",
        "product_name": "MASINGAR Aerosol Product",
        "product_type": "Analysis (ANL)",
        "region": "East Java, Indonesia",
        "spatial_coverage": f"Lat [{cfg['lat_min']}, {cfg['lat_max']}], Lon [{cfg['lon_min']}, {cfg['lon_max']}]",
        "quality_control": (
            "Non-positive and non-finite AOD removed (AOD >= 1e-4); "
            "Filtered strictly for Analysis files ('_ANL' in filename); "
            "Spatial grid consistency strictly enforced via join='exact' (no coordinate override); "
            "Temporal uniqueness strictly verified (script terminates on duplicate timestamps); "
            "Angstrom exponent evaluated for physical boundaries [-0.5, 4.0] prior to turbidity calculation."
        ),
        "wavelengths": "500 nm, 550 nm, 870 nm",
        "notes": (
            "Source dataset is identified as MASINGAR aerosol product (analysis/_ANL). "
            "Classification as reanalysis or operational assimilation is unverified pending dataset documentation confirmation. "
            "Data represents instantaneous values based on filename timestamps. "
            "Ensure 1-day temporal buffer before/after analysis month for rolling 2-point averages."
        ),
    }

    # 8. RINGKASAN STATISTIK DATA
    print("\n" + "=" * 70)
    print(f"RINGKASAN STATISTIK VARIABEL ({ym})")
    print("=" * 70)

    stats_vars = [
        "od500aer",
        "od550aer",
        "od870aer",
        "angstrom_exponent_raw",
        "angstrom_exponent",
        "angstrom_turbidity"
    ]

    stats_rows = []
    total_points = ds_final["od550aer"].size

    for v in stats_vars:
        vals = ds_final[v].values.ravel()
        valid_vals = vals[np.isfinite(vals)]
        n_valid = len(valid_vals)
        n_nan = total_points - n_valid
        pct_valid = (n_valid / total_points * 100.0) if total_points > 0 else 0.0

        if n_valid > 0:
            v_min = float(np.min(valid_vals))
            v_mean = float(np.mean(valid_vals))
            v_std = float(np.std(valid_vals))
            v_median = float(np.median(valid_vals))
            v_max = float(np.max(valid_vals))
        else:
            v_min = v_mean = v_std = v_median = v_max = np.nan

        stats_rows.append({
            "Variabel": v,
            "Total": total_points,
            "Valid": n_valid,
            "NaN": n_nan,
            "% Valid": f"{pct_valid:.1f}%",
            "Min": f"{v_min:.4f}",
            "Mean": f"{v_mean:.4f}",
            "Median": f"{v_median:.4f}",
            "Max": f"{v_max:.4f}",
            "Std": f"{v_std:.4f}",
        })

    stats_df = pd.DataFrame(stats_rows)
    print(stats_df.to_string(index=False))

    # 9. OUTPUT NETCDF
    print("\n" + "=" * 70)
    print("MENYIMPAN OUTPUT NETCDF")
    print("=" * 70)
    os.makedirs(os.path.dirname(output_nc), exist_ok=True)
    ds_final.to_netcdf(output_nc, engine="netcdf4")
    print(f"✓ File NetCDF berhasil disimpan: {output_nc}")

    # 10. OUTPUT TXT
    print("\n" + "=" * 70)
    print("MENYIMPAN OUTPUT TXT")
    print("=" * 70)
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    df_final = ds_final.to_dataframe().reset_index()
    df_final = df_final[
        [
            "valid_time",
            "latitude",
            "longitude",
            "od500aer",
            "od550aer",
            "od870aer",
            "angstrom_exponent_raw",
            "angstrom_exponent",
            "angstrom_turbidity"
        ]
    ]
    df_final["valid_time"] = pd.to_datetime(df_final["valid_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df_final.to_csv(output_txt, sep="\t", index=False)
    print(f"✓ File TXT berhasil disimpan: {output_txt}")

    ds_final.close()
    print(f"\n✓ SELESAI PEMROSESAN BULAN {ym}")


# ============================================================
# MAIN PIPELINE 3 BULAN (1 MUSIM)
# ============================================================

def main():
    print("=" * 70)
    print("PENGOLAHAN DATA AEROSOL MASINGAR (VERSI 8 — 3 BULAN / 1 MUSIM)")
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

        input_dir = INPUT_DIR_TEMPLATE.format(year=year, month=month_2d)

        # Struktur direktori output terpadu: <OUTPUT_BASE>/<TAHUN>/<BULAN_EN>/{nc,txt}/
        output_nc_dir = os.path.join(OUTPUT_BASE, year, month_name, "nc")
        output_txt_dir = os.path.join(OUTPUT_BASE, year, month_name, "txt")
        os.makedirs(output_nc_dir, exist_ok=True)
        os.makedirs(output_txt_dir, exist_ok=True)

        output_nc = os.path.join(output_nc_dir, f"dataaerosol_{ym}.nc")
        output_txt = os.path.join(output_txt_dir, f"dataaerosol_{ym}.txt")

        print(f"\n>>> Memulai Pemrosesan Periode [{idx}/{len(PERIODS)}]: {year}-{month_2d} ({month_name.upper()}) <<<")
        process_aerosol_month(
            year=year,
            month=month_2d,
            input_dir=input_dir,
            output_nc=output_nc,
            output_txt=output_txt,
            cfg=COMMON_CONFIG,
        )

    print("\n" + "=" * 70)
    print("SELURUH PEMROSESAN 3 BULAN AEROSOL SELESAI DENGAN SUKSES!")
    print("=" * 70)


if __name__ == "__main__":
    main()
