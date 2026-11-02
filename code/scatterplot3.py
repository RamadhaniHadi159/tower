import os
import matplotlib
matplotlib.use("Agg")  # Backend non-interaktif, aman untuk server tanpa display GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress


# ============================================================
# KONFIGURASI INPUT & OUTPUT
# ============================================================

# Pilihan Musim yang dianalisis
SEASON = "custom"  # Pilihan: "djf", "mam", "jja", "son", atau "custom"

# Pilihan Form model Csat yang diproses (bisa satu atau keduanya)
# Contoh: ["D"] atau ["D", "E"]
FORMS_TO_PROCESS = ["D", "E"]

# Pilihan Metode yang diproses sekaligus dalam satu kali eksekusi:
# - "set"   : Metode 1 Sel (Nearest Box Masingar)
# - "titik" : Metode 1 Titik (Bilinear Interpolation)
METHODS_TO_PROCESS = ["set", "titik"]

# Direktori pencarian file input masukan (hasil nearest_v8.py)
# Mendukung jalur baru 'ghi/bmkg' dan jalur legacy 'ghi/bmkgjuanda/0_Wm2'
INPUT_DIRS = [
    "/data/mahasiswa/romadhon/perez1/ghi/bmkg",
    "/data/mahasiswa/romadhon/perez1/ghi/bmkgjuanda/0_Wm2",
]

# Stasiun yang akan di-plot (bisa satu atau keduanya)
# Pilihan: ["Juanda"], ["Karangploso"], atau ["Juanda", "Karangploso"]
STATIONS_TO_PLOT = ["Juanda", "Karangploso"]

# Pengaturan Filter QC Linke Turbidity (qc_TL)
# - False : Memplot seluruh data valid tanpa membuang qc_TL = 1 (default)
# - True  : Hanya memplot data dengan Linke Turbidity lolos QC (qc_TL == 0)
FILTER_QC_TL = False

# Direktori dasar penyimpanan grafik output (hanya format PDF)
# Subdirektori otomatis dibuat terpisah per stasiun, per metode, dan per form:
# Struktur: <OUTPUT_GRAPH_BASE>/<stasiun>/<metode>/<form>/
# Contoh:
#   - /data/mahasiswa/romadhon/perez1/graph/bmkg/juanda/set/D/
#   - /data/mahasiswa/romadhon/perez1/graph/bmkg/juanda/set/E/
#   - /data/mahasiswa/romadhon/perez1/graph/bmkg/juanda/titik/D/
#   - /data/mahasiswa/romadhon/perez1/graph/bmkg/juanda/titik/E/
#   - /data/mahasiswa/romadhon/perez1/graph/bmkg/karangploso/set/D/
#   - /data/mahasiswa/romadhon/perez1/graph/bmkg/karangploso/set/E/
#   - /data/mahasiswa/romadhon/perez1/graph/bmkg/karangploso/titik/D/
#   - /data/mahasiswa/romadhon/perez1/graph/bmkg/karangploso/titik/E/
OUTPUT_GRAPH_BASE = "/data/mahasiswa/romadhon/perez1/graph/bmkg"


# ============================================================
# FUNGSI BANTUAN PENCARIAN FILE
# ============================================================

def resolve_input_file(dirs, season, method, form_tag):
    """
    Mencari file input deret waktu masukan (.txt) secara cerdas
    dari daftar direktori yang tersedia.
    """
    filename = f"nearest_{season.lower()}_bmkg_{method.lower()}_form{form_tag.upper()}.txt"
    for d in dirs:
        candidate = os.path.join(d, filename)
        if os.path.isfile(candidate):
            return candidate

    # Default ke direktori pertama jika file belum ada di disk lokal
    return os.path.join(dirs[0], filename)


# ============================================================
# FUNGSI PLOTTING PER STASIUN & METODE
# ============================================================

def generate_scatterplot(df_sub, station_name, season, form_tag, method_tag):
    method_label = "1 Sel" if method_tag.lower() == "set" else "1 Titik"
    station_dir = station_name.lower()
    method_dir = method_tag.lower()
    form_dir = form_tag.upper()

    # Buat direktori khusus per stasiun, per metode, dan per form:
    # <OUTPUT_GRAPH_BASE>/<stasiun>/<metode>/<form>/
    target_out_dir = os.path.join(OUTPUT_GRAPH_BASE, station_dir, method_dir, form_dir)
    os.makedirs(target_out_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"PEMROSESAN SCATTERPLOT: BMKG {station_name.upper()} — Form {form_tag.upper()} ({method_label})")
    print(f"Direktori Target : {target_out_dir}")
    print("=" * 60)

    # 1. Membersihkan data NaN dan Inf
    df_clean = df_sub.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["ssrd(W/m²)", "GHI"]
    ).copy()

    # Opsi filter qc_TL jika diaktifkan
    if FILTER_QC_TL and "qc_TL" in df_clean.columns:
        n_before_qc = len(df_clean)
        df_clean = df_clean[df_clean["qc_TL"] == 0].copy()
        n_after_qc = len(df_clean)
        print(f"Filter qc_TL aktif: {n_after_qc:,} data lolos QC dari {n_before_qc:,} total data.")

    n_samples = len(df_clean)
    print(f"Jumlah observasi valid: {n_samples:,}")
    if n_samples < 2:
        print(f"PERINGATAN: Data untuk stasiun {station_name} ({method_label}) tidak mencukupi untuk regresi.")
        return

    # 2. Variabel Regresi: X = ERA5 SSRD, Y = Perez GHI
    x = df_clean["ssrd(W/m²)"].to_numpy()
    y = df_clean["GHI"].to_numpy()

    # 3. Regresi Linear
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r2 = r_value ** 2

    # 4. Error Metrics
    error = y - x
    mbe = float(np.mean(error))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mean_obs = float(np.mean(x))
    rmbe = (mbe / mean_obs * 100) if mean_obs != 0 else np.nan
    rrmse = (rmse / mean_obs * 100) if mean_obs != 0 else np.nan

    # 5. Cetak Hasil Analisis Statistik
    print(f"Slope (Kemiringan)  = {slope:.4f}")
    print(f"Intercept (Offset)  = {intercept:.4f}")
    print(f"Korelasi (R)        = {r_value:.4f}")
    print(f"Koef. Determinasi R²= {r2:.4f}")
    print(f"P-value             = {p_value:.4e}")
    print(f"MBE                 = {mbe:.4f} W/m² ({rmbe:.2f}%)")
    print(f"RMSE                = {rmse:.4f} W/m² ({rrmse:.2f}%)")

    # 6. Pembuatan Grafik Scatterplot
    plt.figure(figsize=(7, 7))

    x_min, x_max = 0, 1100
    x_line = np.linspace(x_min, x_max, 200)
    y_reg = slope * x_line + intercept

    # Titik scatter
    plt.scatter(
        x,
        y,
        s=25,
        color="black",
        alpha=0.7,
        linewidths=0
    )

    # Garis regresi linear merah (garis putus-putus 1:1 dihilangkan sesuai permintaan)
    plt.plot(
        x_line,
        y_reg,
        color="red",
        linewidth=2
    )

    # Keterangan R² dan Persamaan Regresi pada grafik dengan kotak berlatar belakang bersih
    sign = "+" if intercept >= 0 else "-"
    text = (
        f"$R^2$ = {r2:.3f}\n"
        f"$y$ = {slope:.2f}$x$ {sign} {abs(intercept):.2f}"
    )

    ax = plt.gca()
    # Menggunakan bounding box putih solid agar teks tidak tertimpa titik scatter / garis korelasi
    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="square,pad=0.5",
            facecolor="white",
            edgecolor="gray",
            linewidth=1.0,
            alpha=0.95
        )
    )

    # Label dan Judul
    form_label = f"Form {form_tag.upper()}"
    plt.title(
        f"Global Horizontal Irradiance (GHI)\n"
        f"Semi-Empirical Perez {form_label} (ANNUAL BMKG {station_name} - {method_label})",
        fontsize=13
    )
    # {form_label} or Annual / Custom
    # ({season.upper()} or DJF/MAM/JJA/SON/CUSTOM)
    
    plt.xlabel("ERA5 SSRD (W/m²)", fontsize=13)
    plt.ylabel("Perez GHI (W/m²)", fontsize=13)

    plt.xlim(x_min, x_max)
    plt.ylim(0, 1100)
    plt.grid(False)

    plt.tick_params(direction="in", labelsize=12)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()

    # 7. Simpan Output (Format PDF di direktori khusus stasiun, metode, dan form)
    output_pdf = os.path.join(
        target_out_dir,
        f"scatterplot_ghi_{season.lower()}_bmkg_{station_dir}_{method_dir}_form{form_tag.upper()}.pdf"
    )

    plt.savefig(
        output_pdf,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Grafik PDF berhasil disimpan: {output_pdf}")


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main():
    print("=" * 70)
    print("PROGRAM PEMBUATAN SCATTERPLOT GHI VS ERA5 SSRD (SCATTERPLOT V6)")
    print("=" * 70)
    print(f"Musim Diproses     : {SEASON.upper()}")
    print(f"Form Diproses      : {FORMS_TO_PROCESS} (D dan E)")
    print(f"Metode Diproses    : {METHODS_TO_PROCESS} (1 Sel & 1 Titik)")
    print(f"Stasiun Diproses   : {STATIONS_TO_PLOT}")
    print(f"Filter qc_TL       : {FILTER_QC_TL}")
    print(f"Direktori Basis    : {OUTPUT_GRAPH_BASE}")
    print("Struktur Folder    : <Basis>/<stasiun>/<metode>/<form>/")
    print("=" * 70)

    total_grafik = 0

    for form in FORMS_TO_PROCESS:
        for method in METHODS_TO_PROCESS:
            method_name = "1 Sel (Nearest Box)" if method == "set" else "1 Titik (Bilinear)"
            file_input = resolve_input_file(INPUT_DIRS, SEASON, method, form)

            print("\n" + "#" * 70)
            print(f"MEMPROSES DATA: Form {form.upper()} — Metode {method_name}")
            print(f"File Masukan   : {file_input}")
            print("#" * 70)

            if not os.path.isfile(file_input):
                print(f"PERINGATAN: File input tidak ditemukan: {file_input}")
                print(f"Pastikan skrip nearest_v8.py telah dijalankan untuk form {form.upper()} ({method}).")
                continue

            # Baca file deret waktu (hasil nearest_v8.py)
            df = pd.read_csv(file_input, sep="\t")
            print(f"Total baris data dibaca: {len(df):,}")

            if "station" not in df.columns:
                print(f"ERROR: Kolom 'station' tidak ditemukan pada file {file_input}.")
                continue

            # Proses pembuatan scatterplot per stasiun yang dipilih
            for station in STATIONS_TO_PLOT:
                df_station = df[df["station"] == station].copy()
                if df_station.empty:
                    print(f"PERINGATAN: Tidak ada data untuk stasiun '{station}' di dalam file.")
                    continue

                generate_scatterplot(
                    df_sub=df_station,
                    station_name=station,
                    season=SEASON,
                    form_tag=form,
                    method_tag=method,
                )
                total_grafik += 1

    print("\n" + "=" * 70)
    print(f"SELURUH PROSES SELESAI: Total {total_grafik} grafik scatterplot PDF berhasil dibuat!")
    print(f"Grafik tersimpan terpisah pada folder masing-masing stasiun, metode, dan form di:")
    print(f"  {OUTPUT_GRAPH_BASE}/<stasiun>/<metode>/<form>/")
    print("=" * 70)


if __name__ == "__main__":
    main()
