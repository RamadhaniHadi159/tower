import calendar
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# KONFIGURASI PENGGABUNGAN (SELARAS DENGAN perez3_v12.py)
# ============================================================

# Direktori dasar output dari perez3_v12.py
INPUT_BASE = "/data/mahasiswa/romadhon/perez1/ghi"

# Direktori dasar output hasil penggabungan (combined)
OUTPUT_BASE = "/data/mahasiswa/romadhon/perez1/ghi/combined"

# Pilihan Musim yang Diproses
# Nilai yang didukung: "DJF", "MAM", "JJA", "SON", atau "CUSTOM"
SEASON = "SON"  # Ganti sesuai kebutuhan

# Definisi periode per musim (format: [(tahun, bulan_2d), ...])
# Sesuai dengan konfigurasi musim pada perez3_v12.py
SEASON_DEFINITIONS = {
    "DJF": [("2021", "12"), ("2022", "01"), ("2022", "02")],
    "MAM": [("2022", "03"), ("2022", "04"), ("2022", "05")],
    "JJA": [("2022", "06"), ("2022", "07"), ("2022", "08")],
    "SON": [("2022", "09"), ("2022", "10"), ("2022", "11")],
}

# Periode custom jika SEASON == "CUSTOM"
CUSTOM_PERIODS = [
    ("2021", "12"), ("2022", "01"), ("2022", "02"),                                                                           
    ("2022", "03"), ("2022", "04"), ("2022", "05"),                                                                           
    ("2022", "06"), ("2022", "07"), ("2022", "08"),                                                                           
    ("2022", "09"), ("2022", "10"), ("2022", "11"),
]

# Bentuk Form Csat yang diproses:
# Menghasilkan tepat 2 output terpisah: Form D dan Form E
FORMS_TO_PROCESS = ["D", "E"]


# ============================================================
# DEFINISI ATRIBUT & METADATA VARIABEL (SESUAI perez3_v12.py)
# ============================================================

ATTRS = {
    "GHI": ("Global Horizontal Irradiance", "W m-2"),
    "GHIC": ("Clear-sky Global Horizontal Irradiance", "W m-2"),
    "Csat": ("Normalized satellite reflectance (Perez)", "1"),
    "CI": ("Cloud Index", "1"),
    "TL": ("Linke Turbidity", "1"),
    "qc_TL": ("QC flag Linke Turbidity > TL_MAX", "1"),
    "kc": ("Clear-sky index (Perez Ktm)", "1"),
    "cmin": ("Lower bound of Csat", "1"),
    "cmax": ("Upper bound of Csat", "1"),
    "reflectance": ("Reflectance", "1"),
    "am": ("Air mass", "1"),
    "epsilon": ("Earth-Sun distance correction", "1"),
    "sza_deg": ("Solar zenith angle", "degree"),
    "tcwv_cm": ("Total column water vapour", "cm"),
    "angstrom_turbidity": ("Angstrom turbidity beta", "1"),
    "elevation_m": ("Elevation", "m"),
    "I0_Wm2": ("Extraterrestrial irradiance", "W m-2"),
    "ssrd_Wm2": ("ERA5 surface solar radiation downwards", "W m-2"),
    "cosz_mean": ("Mean cosine of solar zenith angle over the hour", "1"),
}

# Kolom wajib yang harus ada di dalam dataset
REQUIRED_COLUMNS = [
    "valid_time",
    "latitude",
    "longitude",
    "angstrom_turbidity",
    "reflectance",
    "sza_deg",
    "am",
    "epsilon",
    "I0_Wm2",
    "elevation_m",
    "tcwv_cm",
    "ssrd_Wm2",
    "Csat",
    "CI",
    "TL",
    "qc_TL",        # Wajib disertakan untuk tahap downstream (nearest_v4)
    "GHIC",
    "kc",
    "GHI",
]

# Pemetaan nama kolom untuk file output TXT
TXT_COLUMN_MAPPING = {
    "valid_time": "valid_time",
    "latitude": "latitude",
    "longitude": "longitude",
    "angstrom_turbidity": "angstrom_turbidity(dimensionless)",
    "reflectance": "reflectance(dimensionless)",
    "sza_deg": "sza(°)",
    "am": "am(dimensionless)",
    "epsilon": "epsilon(dimensionless)",
    "I0_Wm2": "I0(W/m²)",
    "elevation_m": "elevation(m)",
    "tcwv_cm": "tcwv(cm)",
    "ssrd_Wm2": "ssrd(W/m²)",
    "Csat": "Csat",
    "CI": "CI",
    "TL": "TL",
    "qc_TL": "qc_TL",
    "GHIC": "GHIC",
    "kc": "kc",
    "GHI": "GHI",
}


# ============================================================
# FUNGSI BANTUAN PENCARIAN FILE INPUT
# ============================================================

def resolve_input_nc(input_base, year, month_str, form_tag):
    """
    Mencari lokasi file NetCDF keluaran perez3_v12.py secara cerdas.
    Mendukung format folder resmi perez3_v12 (dengan/tanpa form D/E)
    serta jalur legacy.
    """
    year_str = str(year)
    month_2d = str(month_str).zfill(2)
    month_name = calendar.month_name[int(month_str)].lower()

    candidates = [
        # 1. Output resmi perez3_v12: <BASE>/<TAHUN>/<BULAN_EN>/<FORM>/nc/ghi_<TAHUN>-<BULAN_2D>.nc
        os.path.join(input_base, year_str, month_name, form_tag, "nc", f"ghi_{year_str}-{month_2d}.nc"),
        # 2. Output perez3_v12 tanpa subfolder form: <BASE>/<TAHUN>/<BULAN_EN>/nc/ghi_<TAHUN>-<BULAN_2D>.nc
        os.path.join(input_base, year_str, month_name, "nc", f"ghi_{year_str}-{month_2d}.nc"),
        # 3. Output dengan format nama bulan teks: ghi_<BULAN_EN>.nc
        os.path.join(input_base, year_str, month_name, form_tag, "nc", f"ghi_{month_name}.nc"),
        os.path.join(input_base, year_str, month_name, "nc", f"ghi_{month_name}.nc"),
        # 4. Format legacy path (combined3 lama): <BASE>/0_Wm2/nc/ghi_<BULAN_EN>.nc
        os.path.join(input_base, "0_Wm2", "nc", f"ghi_{month_name}.nc"),
        os.path.join(input_base, "0_Wm2", "nc", f"ghi_{year_str}-{month_2d}.nc"),
    ]

    for cand in candidates:
        if os.path.isfile(cand):
            return cand

    # Default ke candidate 1 jika belum dijalankan di disk lokal
    return candidates[0]


# ============================================================
# PEMROSESAN PENGGABUNGAN DATA (COMBINE)
# ============================================================

def process_combine_season(periods, season_name, form_tag):
    season_lower = season_name.lower()
    form_label = f"form{form_tag.upper()}"
    form_desc = "Perez et al. (2002)" if form_tag.upper() == "D" else "refl_cos"

    print("\n" + "=" * 76)
    print(f"PENGGABUNGAN DATA GHI — MUSIM {season_name.upper()} ({form_label} - {form_desc})")
    print("=" * 76)

    # 1. Tentukan dan verifikasi file input
    input_files = []
    print("Mempersiapkan file input dari perez3_v12.py:")
    for yr, mo in periods:
        path = resolve_input_nc(INPUT_BASE, yr, mo, form_tag)
        input_files.append((yr, mo, path))
        print(f"  - Periode {yr}-{str(mo).zfill(2)} : {path}")

    # 2. Baca seluruh dataset bulanan
    df_list = []
    print("\nMembaca dataset bulanan...")
    for yr, mo, file_path in input_files:
        print(f"  Membaca: {file_path}")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"File input tidak ditemukan: {file_path}\n"
                f"Pastikan perez3_v12.py telah dieksekusi untuk periode {yr}-{mo} ({form_label})."
            )

        ds = xr.open_dataset(file_path)
        df_month = ds.to_dataframe().reset_index()

        # Bersihkan index internal obs jika ada
        if "obs" in df_month.columns:
            df_month = df_month.drop(columns=["obs"])

        print(f"    -> Jumlah observasi: {len(df_month):,}")
        df_list.append(df_month)
        ds.close()

    # 3. Gabungkan seluruh data bulanan
    print("\nPenggabungan data frame...")
    df_merged = pd.concat(df_list, ignore_index=True)
    print(f"Total baris gabungan sebelum pengurutan: {len(df_merged):,}")

    # 4. Pastikan valid_time bertipe datetime dan urutkan secara kronologis
    if "valid_time" in df_merged.columns:
        df_merged["valid_time"] = pd.to_datetime(df_merged["valid_time"], errors="coerce")
        df_merged = df_merged.sort_values(by="valid_time").reset_index(drop=True)

    # 5. Validasi ketersediaan kolom wajib (termasuk qc_TL)
    missing = [c for c in REQUIRED_COLUMNS if c not in df_merged.columns]
    if missing:
        print(f"\nERROR: Variabel wajib berikut tidak ditemukan dalam dataset:")
        for m in missing:
            print(f"  - {m}")
        print(f"Variabel yang tersedia dalam dataset: {df_merged.columns.tolist()}")
        raise KeyError(f"Variabel wajib tidak lengkap: {missing}")

    # Susun kolom: kolom wajib terlebih dahulu, lalu kolom tambahan (seperti bounds_src, cmin, cmax)
    extra_cols = [c for c in df_merged.columns if c not in REQUIRED_COLUMNS and c != "ym"]
    final_cols = REQUIRED_COLUMNS + extra_cols
    df_merged = df_merged[final_cols].copy()

    # 6. Susun xarray Dataset terstandar (identik dengan struktur perez3_v12.py)
    data_vars = {}
    for col, (long_name, units) in ATTRS.items():
        if col in df_merged.columns:
            # Gunakan float32 untuk data numerik seperti perez3_v12.py
            data_vars[col] = (
                ["obs"],
                df_merged[col].to_numpy(dtype="float32"),
                {"long_name": long_name, "units": units}
            )

    if "bounds_src" in df_merged.columns:
        data_vars["bounds_src"] = (["obs"], df_merged["bounds_src"].astype(str).to_numpy())

    ds_merged = xr.Dataset(
        data_vars,
        coords={
            "obs": np.arange(len(df_merged)),
            "valid_time": ("obs", pd.to_datetime(df_merged["valid_time"]).to_numpy()),
            "latitude": ("obs", df_merged["latitude"].to_numpy(), {"units": "degrees_north"}),
            "longitude": ("obs", df_merged["longitude"].to_numpy(), {"units": "degrees_east"}),
        },
        attrs={
            "title": f"Combined Seasonal GHI ({season_name.upper()}) - Perez et al. (2002) model",
            "season": season_name.upper(),
            "form": form_label,
            "csat_form": form_label,
            "region": "Jawa Timur",
            "source": "Himawari-8 B03, MASINGAR AOD, ERA5 TCWV, COP30 DEM",
        },
    )

    # 7. Tentukan path output tunggal per form (Hanya 1 file NC dan 1 file TXT per form)
    # Output:
    # - Form D : nc/ghi_<season>_season_formD.nc
    # - Form E : nc/ghi_<season>_season_formE.nc
    nc_dir = os.path.join(OUTPUT_BASE, "nc")
    txt_dir = os.path.join(OUTPUT_BASE, "txt")
    os.makedirs(nc_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)

    nc_file = os.path.join(nc_dir, f"ghi_{season_lower}_season_{form_label}.nc")
    txt_file = os.path.join(txt_dir, f"ghi_{season_lower}_season_{form_label}.txt")

    # Simpan NetCDF
    if os.path.exists(nc_file):
        os.remove(nc_file)
    ds_merged.to_netcdf(nc_file, format="NETCDF4")
    print(f"\nNetCDF berhasil disimpan : {nc_file}")
    ds_merged.close()

    # Simpan TXT
    df_txt = df_merged.copy()
    df_txt = df_txt.rename(columns=TXT_COLUMN_MAPPING)

    if os.path.exists(txt_file):
        os.remove(txt_file)
    df_txt.to_csv(txt_file, sep="\t", index=False, float_format="%.6g")
    print(f"TXT berhasil disimpan    : {txt_file}")

    # 8. Laporan Ringkasan
    print("\n" + "-" * 76)
    print(f"RINGKASAN HASIL PENGGABUNGAN ({season_name.upper()} - {form_label})")
    print("-" * 76)
    print(f"Total baris data : {len(df_merged):,}")
    print(f"Jumlah kolom     : {len(df_merged.columns)}")
    print(f"Rentang waktu    : {df_merged['valid_time'].min()} s.d. {df_merged['valid_time'].max()}")
    print("Daftar variabel NetCDF:")
    for i, col in enumerate(df_merged.columns, 1):
        flag = " [KOLOM WAJIB]" if col in REQUIRED_COLUMNS else ""
        print(f"  {i:2d}. {col:20s}{flag}")


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main():
    print("=" * 76)
    print("PROGRAM PENGGABUNGAN DATA GHI MUSIMAN (COMBINED V6)")
    print("=" * 76)

    # Tentukan periode bulan berdasarkan musim
    if SEASON.upper() in SEASON_DEFINITIONS:
        periods = SEASON_DEFINITIONS[SEASON.upper()]
        season_name = SEASON.upper()
    else:
        periods = CUSTOM_PERIODS
        season_name = "CUSTOM"

    print(f"Musim Dipilih    : {season_name}")
    print(f"Periode Bulan    : {periods}")
    print(f"Form Diproses    : {FORMS_TO_PROCESS}")
    print(f"Input Base       : {INPUT_BASE}")
    print(f"Output Base      : {OUTPUT_BASE}")

    for form in FORMS_TO_PROCESS:
        process_combine_season(
            periods=periods,
            season_name=season_name,
            form_tag=form,
        )

    print("\n" + "=" * 76)
    print("SELURUH PROSES PENGGABUNGAN DATA BERHASIL DISELESAIKAN!")
    print("=" * 76)


if __name__ == "__main__":
    main()
