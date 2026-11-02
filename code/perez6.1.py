import calendar
import os

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

# Template Path File Input NetCDF (Keluaran regrid10)
# Variabel {year}, {month}, dan {month_name} disubstitusikan untuk setiap bulan di PERIODS.
INPUT_NETCDF_TEMPLATE = (
    "/data/mahasiswa/romadhon/perez1/perez/{year}/"
    "{month_name}/nc/perez_{year}{month}.nc"
)

# Direktori Output Dasar (Tersatu)
# Subfolder per bulan otomatis: <OUTPUT_BASE>/<TAHUN>/<BULAN_EN>/{D,E}/{nc,txt}/
OUTPUT_BASE = "/data/mahasiswa/romadhon/perez1/ghi"

# Parameter Model & Ambang Batas QC
TL_MAX = 10.0             # batas atas Linke Turbidity untuk QC TL
SZA_MAX = 80.0            # batas sudut zenit (Gueymard & Ruiz-Arias, 2016)
H_MIN, H_MAX = 1.5, 65.0  # batas elevasi matahari pada normalisasi Perez (derajat)
MIN_SAMPLES = 30          # minimal sampel per sel per bulan untuk Cmin/Cmax per sel
N_HIGH = 10               # Cmax = rerata N_HIGH nilai tertinggi
LOW_SLICE = slice(1, 5)   # Cmin = rerata nilai terendah ke-2 s.d. ke-5
TL_B_W2 = 0.0254          # koefisien W^2 pada suku beta (Remund dkk., 2003)
CELL = ["latitude", "longitude"]

DEFAULT_CSAT_FORM = "refl_cos"
DEFAULT_CI_METHOD = "chen_cell"
RUN_SENSITIVITY = True
RUN_ALL_FORMS = True       # True = jalankan perez (D) + refl_cos (E) sekaligus
SAVE_SZA_BIN_TABLE = True

REQUIRED = [
    "valid_time", "latitude", "longitude", "reflectance", "sza_deg", "am",
    "epsilon", "I0_Wm2", "elevation_m", "tcwv_cm", "angstrom_turbidity",
]


# ----------------------------------------------------------------------
# FUNGSI MODEL
# ----------------------------------------------------------------------
# Reflectance Normalitation (C_sat), Perez et al. (2002)
def csat_perez(df, form="perez", use_eps=True):
    """Normalisasi reflektansi sesuai rumus Perez (model acuan pengguna)."""
    c_raw = df["reflectance"]
    am = df["am"]
    eps = df["epsilon"] if use_eps else 1.0

    if form == "perez":
        h = np.clip(90.0 - df["sza_deg"], H_MIN, H_MAX)
        return (c_raw * am * eps) / (
            2.283 * h ** (-0.26) * np.exp(0.004 * h)
        )

    if form == "refl_cos":
        h = np.clip(90.0 - df["sza_deg"], H_MIN, H_MAX)
        return c_raw / (2.283 * h ** (-0.26) * np.exp(0.004 * h))

    if form == "legacy":
        # Hanya untuk pembanding sensitivitas; bukan rumus utama.
        h = 90.0 - df["sza_deg"]
        return (c_raw * am * eps / 2.283) * h ** (-0.26) * np.exp(0.004 * h)

    raise ValueError(f"Metode Csat tidak dikenal: {form}")


# Cloud index (CI), Chen dkk. (2010) per sel grid 0.05° x 0.05°.
def chen_bounds(x):
    s = np.sort(np.asarray(x, dtype=float))
    return pd.Series({
        "cmin": s[LOW_SLICE].mean(),
        "cmax": s[-N_HIGH:].mean(),
        "n": len(s),
    })


def cloud_index(df, method="chen_cell"):
    """CI = (Csat - Cmin) / (Cmax - Cmin), dibatasi [0,1]."""
    df = df.copy()
    df["ym"] = pd.to_datetime(df["valid_time"]).dt.to_period("M")

    if method == "percentile":  # pembanding: P1/P99 seluruh domain per bulan
        g = df.groupby("ym")["Csat"]
        df["cmin"] = g.transform(lambda s: np.percentile(s, 1))
        df["cmax"] = g.transform(lambda s: np.percentile(s, 99))
        df["bounds_src"] = "percentile"
    elif method == "chen_cell":
        cell = df.groupby(["ym"] + CELL)["Csat"].apply(chen_bounds).unstack()
        dom = df.groupby("ym")["Csat"].apply(chen_bounds).unstack()
        cell = cell.join(dom, on="ym", rsuffix="_dom")
        ok = cell["n"] >= MIN_SAMPLES
        cell["cmin_use"] = np.where(ok, cell["cmin"], cell["cmin_dom"])
        cell["cmax_use"] = np.where(ok, cell["cmax"], cell["cmax_dom"])
        cell["bounds_src"] = np.where(ok, "cell", "domain")
        df = df.join(cell[["cmin_use", "cmax_use", "bounds_src", "n"]], on=["ym"] + CELL)
        df = df.rename(columns={"cmin_use": "cmin", "cmax_use": "cmax", "n": "n_cell"})
    else:
        raise ValueError(method)

    span = df["cmax"] - df["cmin"]
    df["CI"] = ((df["Csat"] - df["cmin"]) / span).where(span > 0).clip(0, 1)
    return df


# Linke turbidity (TL), Remund dkk. (2003)
def linke_turbidity(df, w_div10=False):
    """TL (Remund dkk., 2003). W = uap air terpresipitasi dalam cm."""
    w = df["tcwv_cm"] / 10.0 if w_div10 else df["tcwv_cm"]  # w_div10=True hanya utk sensitivitas
    beta = df["angstrom_turbidity"]
    return (1.8494 + 0.2425 * w - 0.0203 * w**2) + (15.427 + 0.3153 * w - TL_B_W2 * w**2) * beta


# Global horizontal irradiance Clear-sky (GHIC), Ineichen & Perez (2002)
def clear_sky_ghi(df):
    """GHIC mengikuti formulasi Ineichen & Perez (2002) pada kode acuan."""
    z, am, tl, i0 = (
        df["elevation_m"], df["am"], df["TL"], df["I0_Wm2"]
    )
    cosz = np.cos(np.radians(df["sza_deg"]))
    cg1 = 0.0000509 * z + 0.868
    cg2 = 0.0000392 * z + 0.0387
    fh1 = np.exp(-z / 8000.0)
    fh2 = np.exp(-z / 1250.0)
    return (
        cg1 * i0 * cosz
        * np.exp(-cg2 * am * (fh1 + fh2 * (tl - 1)))
        * np.exp(0.01 * am**1.8)
    )


# Global horizontal irradiance All-sky (GHI), Perez et al. (2002)
def all_sky_ghi(df):
    ci = df["CI"]
    kc = 2.36 * ci**5 - 6.2 * ci**4 + 6.22 * ci**3 - 2.63 * ci**2 - 0.58 * ci + 1
    ghic = df["GHIC"]
    return kc, kc * ghic * (0.0001 * kc * ghic + 0.9)


def run_model(df, csat_form="perez", ci_method="chen_cell", w_div10=False, use_eps=True):
    df = df.copy()
    df["Csat"] = csat_perez(df, csat_form, use_eps)
    df = df[np.isfinite(df["Csat"])].copy()
    df = cloud_index(df, ci_method)
    df["TL"] = linke_turbidity(df, w_div10)
    df["qc_TL"] = (linke_turbidity(df, w_div10=False) > TL_MAX).astype(int)
    df["GHIC"] = clear_sky_ghi(df)
    df["kc"], df["GHI"] = all_sky_ghi(df)
    cols = ["Csat", "CI", "TL", "GHIC", "kc", "GHI"]
    ok = np.isfinite(df[cols]).all(axis=1)
    return df[ok].reset_index(drop=True), int((~ok).sum())


# ----------------------------------------------------------------------
# QC DAN METRIK
# ----------------------------------------------------------------------
def load_and_qc(path, month=None):
    """
    Membaca data NetCDF dari regrid10 dan menerapkan QC input.
    Catatan: Tanpa penyaringan jam pengamatan (semua jam dalam file diproses).
    """
    with xr.open_dataset(path) as ds:
        df = ds.to_dataframe().reset_index()

    # Batasi rentang waktu sesuai month jika diberikan (format YYYY-MM).
    df["valid_time"] = pd.to_datetime(df["valid_time"])
    if month:
        month_start = pd.Period(month, freq="M").start_time
        month_end = (pd.Period(month, freq="M") + 1).start_time
        before = len(df)
        df = df[
            (df["valid_time"] >= month_start) &
            (df["valid_time"] < month_end)
        ].copy()
        print(f"Baris di luar bulan {month} dibuang: {before - len(df)}")

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"Kolom wajib tidak ada: {missing}")

    n0 = len(df)
    num = ["reflectance", "sza_deg", "am", "epsilon", "I0_Wm2",
           "elevation_m", "tcwv_cm", "angstrom_turbidity"]
    ok = np.isfinite(df[num]).all(axis=1) & (df["sza_deg"] >= 0) & (df["sza_deg"] < SZA_MAX)
    print(f"Baris awal            : {n0}")
    print(f"Dibuang (NaN/sza>=80) : {int((~ok).sum())}")
    df = df[ok].reset_index(drop=True)
    print(f"Baris dipakai         : {len(df)}")

    # pemeriksaan satuan TCWV: Jawa Timur lembap, W wajar 2-7 cm (20-70 kg m-2)
    med = df["tcwv_cm"].median()
    print(f"tcwv_cm  min/median/max: {df['tcwv_cm'].min():.2f} / {med:.2f} / {df['tcwv_cm'].max():.2f}")
    if not (1.5 <= med <= 8.0):
        print("PERINGATAN: median tcwv_cm di luar 1,5-8 cm. Periksa satuan (mm atau cm?).")
    return df


def metrics(obs, mod):
    obs, mod = np.asarray(obs, float), np.asarray(mod, float)
    m = np.isfinite(obs) & np.isfinite(mod)
    obs, mod = obs[m], mod[m]

    if len(obs) == 0:
        return {
            "n": 0, "RMSE": np.nan, "rRMSE_%": np.nan,
            "MBE": np.nan, "rMBE_%": np.nan, "R2": np.nan,
        }

    d = mod - obs
    rmse = float(np.sqrt(np.mean(d**2)))
    mbe = float(np.mean(d))
    if len(obs) >= 2 and np.std(obs) > 0 and np.std(mod) > 0:
        r2 = float(np.corrcoef(obs, mod)[0, 1] ** 2)
    else:
        r2 = np.nan
    mean_obs = float(np.mean(obs))

    return {
        "n": len(obs),
        "RMSE": rmse,
        "rRMSE_%": 100 * rmse / mean_obs if mean_obs != 0 else np.nan,
        "MBE": mbe,
        "rMBE_%": 100 * mbe / mean_obs if mean_obs != 0 else np.nan,
        "R2": r2,
    }


def sensitivity(df_in):
    if "ssrd_Wm2" not in df_in.columns:
        print("Kolom ssrd_Wm2 tidak tersedia; sensitivitas ERA5 dilewati.")
        return None

    # Skenario pembanding seperti kode model acuan.
    runs = {
        "A. Kode lama (legacy + P1/P99 + W/10)": dict(
            csat_form="legacy", ci_method="percentile", w_div10=True
        ),
        "B. Csat Perez + P1/P99 + W/10": dict(
            csat_form="perez", ci_method="percentile", w_div10=True
        ),
        "C. Perez + P1/P99 + TCWV cm langsung": dict(
            csat_form="perez", ci_method="percentile", w_div10=False
        ),
        "D. Perez + Chen per sel + TCWV cm langsung (utama)": dict(
            csat_form="perez", ci_method="chen_cell", w_div10=False
        ),
        "E. refl_cos (tanpa AM/eps) + Chen per sel + TCWV cm": dict(
            csat_form="refl_cos", ci_method="chen_cell", w_div10=False
        ),
    }

    rows = {}
    for name, kw in runs.items():
        out, _ = run_model(df_in, **kw)
        if out.empty:
            rows[name] = metrics([], [])
            continue
        valid = np.isfinite(out["ssrd_Wm2"]) & (out["ssrd_Wm2"] > 0)
        if "qc_TL" in out.columns:
            valid = valid & (out["qc_TL"] == 0)
        rows[name] = metrics(
            out.loc[valid, "ssrd_Wm2"], out.loc[valid, "GHI"]
        )

    table = pd.DataFrame(rows).T
    print("\n===== SENSITIVITAS TERHADAP ERA5 SSRD (qc_TL == 0) =====")
    print(table.round(3).to_string())
    return table


def save_sza_bin_table(df, outdir, month):
    """Simpan ringkasan Csat menurut bin SZA untuk QC."""
    if df.empty:
        print("Tabel bin SZA dilewati: data kosong.")
        return None
    work = df[["sza_deg", "reflectance", "am", "Csat"]].copy()
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 75, 80]
    labels = ["0-10", "10-20", "20-30", "30-40", "40-50",
              "50-60", "60-70", "70-75", "75-80"]
    work["sza_bin_deg"] = pd.cut(work["sza_deg"], bins=bins, labels=labels,
                                  right=False, include_lowest=True)
    table = work.groupby("sza_bin_deg", observed=False).agg(
        n=("Csat", "count"), sza_mean=("sza_deg", "mean"),
        reflectance_mean=("reflectance", "mean"), am_mean=("am", "mean"),
        Csat_mean=("Csat", "mean"), Csat_median=("Csat", "median"),
        Csat_max=("Csat", "max")
    ).reset_index()
    path = os.path.join(outdir, f"csat_sza_bins_{month}.csv")
    table.to_csv(path, index=False, float_format="%.6g")
    print("\n===== STATISTIK CSAT PER BIN SZA =====")
    print(table.round(4).to_string(index=False))
    print("Output tabel bin SZA:", path)
    return table


# ----------------------------------------------------------------------
# OUTPUT
# ----------------------------------------------------------------------
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


def save(df, outdir, month):
    os.makedirs(os.path.join(outdir, "nc"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "txt"), exist_ok=True)
    txt = os.path.join(outdir, "txt", f"ghi_{month}.txt")
    nc = os.path.join(outdir, "nc", f"ghi_{month}.nc")

    df.drop(columns=["ym"], errors="ignore").to_csv(txt, sep="\t", index=False, float_format="%.6g")

    data_vars = {}
    for col, (long_name, units) in ATTRS.items():
        if col in df.columns:
            data_vars[col] = (["obs"], df[col].to_numpy(dtype="float32"),
                              {"long_name": long_name, "units": units})
    data_vars["bounds_src"] = (["obs"], df["bounds_src"].astype(str).to_numpy())
    ds = xr.Dataset(
        data_vars,
        coords={
            "obs": np.arange(len(df)),
            "valid_time": ("obs", pd.to_datetime(df["valid_time"]).to_numpy()),
            "latitude": ("obs", df["latitude"].to_numpy(), {"units": "degrees_north"}),
            "longitude": ("obs", df["longitude"].to_numpy(), {"units": "degrees_east"}),
        },
        attrs={
            "title": "GHI estimated with the Perez et al. (2002) semi-empirical model",
            "region": "Jawa Timur",
            "source": "Himawari-8 B03, MASINGAR AOD, ERA5 TCWV, COP30 DEM",
        },
    )
    ds.to_netcdf(nc, format="NETCDF4")
    print(f"\nOutput TXT: {txt}\nOutput NC : {nc}")


# ----------------------------------------------------------------------
# PEMROSESAN MODEL PER BULAN
# ----------------------------------------------------------------------
def process_perez_month(input_nc, month_str, outdir_perez, outdir_refl_cos):
    print("=" * 72)
    print(f"MODEL SEMI-EMPIRIS PEREZ — BULAN {month_str}")
    print("=" * 72)
    print("Input             :", input_nc)
    print("Label bulan       :", month_str)
    print("Metode CI         :", DEFAULT_CI_METHOD)
    print("Sensitivitas      :", RUN_SENSITIVITY)
    print("Mode              :", "all-forms (perez + refl_cos)" if RUN_ALL_FORMS else DEFAULT_CSAT_FORM)
    print("Output Perez (D)  :", outdir_perez)
    print("Output refl_cos(E):", outdir_refl_cos)
    print("Catatan           : Tanpa filter jam pengamatan (semua jam diproses)")

    if not os.path.isfile(input_nc):
        raise FileNotFoundError(f"File input tidak ditemukan: {input_nc}")

    df = load_and_qc(input_nc, month=month_str)
    if df.empty:
        raise ValueError(f"Tidak ada data valid setelah QC input untuk bulan {month_str}.")

    print("Cosine untuk GHIC : cos(sza_deg) sesuai kode model acuan")

    form_outdirs = {
        "perez": outdir_perez,
        "refl_cos": outdir_refl_cos,
    }
    forms_to_run = list(form_outdirs.keys()) if RUN_ALL_FORMS else [DEFAULT_CSAT_FORM]

    for form in forms_to_run:
        outdir = form_outdirs.get(form, outdir_refl_cos)
        os.makedirs(outdir, exist_ok=True)

        print("\n" + "=" * 72)
        print(f"FORM CSAT : {form}   |   OUTPUT : {outdir}")
        print("=" * 72)

        out, n_bad = run_model(df, csat_form=form, ci_method=DEFAULT_CI_METHOD,
                                w_div10=False, use_eps=True)
        print(f"\nHasil model tidak valid dibuang: {n_bad}")
        print(f"Jumlah data final              : {len(out)}")
        if out.empty:
            print(f"PERINGATAN: Tidak ada data valid untuk form '{form}'. Dilewati.")
            continue

        if "bounds_src" in out:
            print("\nSumber Cmin/Cmax:")
            print(out["bounds_src"].value_counts().to_string())

        print("\n===== STATISTIK MODEL PEREZ =====")
        print(out[["Csat", "CI", "TL", "GHIC", "kc", "GHI"]].describe().round(3).to_string())

        if "ssrd_Wm2" in out.columns:
            valid = np.isfinite(out["ssrd_Wm2"]) & (out["ssrd_Wm2"] > 0)
            if "qc_TL" in out.columns:
                valid = valid & (out["qc_TL"] == 0)
            result = metrics(out.loc[valid, "ssrd_Wm2"], out.loc[valid, "GHI"])
            print("\n===== VALIDASI ERA5 SSRD (qc_TL == 0) =====")
            for key, value in result.items():
                print(f"{key}: {value:.4f}" if pd.notna(value) else f"{key}: NaN")
        else:
            print("\nKolom ssrd_Wm2 tidak tersedia; validasi ERA5 dilewati.")

        if RUN_SENSITIVITY:
            tab = sensitivity(df)
            if tab is not None:
                path = os.path.join(outdir, f"sensitivity_{month_str}.csv")
                tab.to_csv(path, float_format="%.6g")
                print("Output sensitivitas:", path)

        if SAVE_SZA_BIN_TABLE:
            qc_csat = df.copy()
            qc_csat["Csat"] = csat_perez(qc_csat, form=form, use_eps=True)
            qc_csat = qc_csat[np.isfinite(qc_csat["Csat"])].copy()
            save_sza_bin_table(qc_csat, outdir, month_str)

        save(out, outdir, month_str)

    print(f"\nPemrosesan bulan {month_str} selesai.")


# ----------------------------------------------------------------------
# MAIN PIPELINE 3 BULAN (1 MUSIM)
# ----------------------------------------------------------------------
def main():
    print("=" * 72)
    print("MODEL SEMI-EMPIRIS PEREZ — FIXED V12 (3 BULAN / 1 MUSIM)")
    print("=" * 72)
    print(f"Jumlah Periode Bulan : {len(PERIODS)} bulan")
    for idx, (yr, mo) in enumerate(PERIODS, 1):
        print(f"  Bulan ke-{idx}          : Tahun {yr}, Bulan {str(mo).zfill(2)}")
    print(f"Direktori Basis Out  : {OUTPUT_BASE}")
    print("=" * 72)

    for idx, (year_str, month_str) in enumerate(PERIODS, 1):
        year = str(year_str)
        month_2d = str(month_str).zfill(2)
        month_name = calendar.month_name[int(month_str)].lower()
        month_label = f"{year}-{month_2d}"
        ym = f"{year}{month_2d}"

        input_nc = INPUT_NETCDF_TEMPLATE.format(
            year=year, month=month_2d, month_name=month_name
        )
        # Fallback jika data regrid10 berada pada format folder lama (monthly/nc/)
        if not os.path.isfile(input_nc):
            legacy_input = f"/data/mahasiswa/romadhon/perez1/perez/monthly/nc/perez_{ym}.nc"
            if os.path.isfile(legacy_input):
                input_nc = legacy_input

        # Subfolder otomatis berdasarkan tahun dan nama bulan bahasa Inggris
        # Jika RUN_ALL_FORMS=True, subfolder D dan E digunakan agar kedua form tidak saling menimpa
        if RUN_ALL_FORMS:
            outdir_perez = os.path.join(OUTPUT_BASE, year, month_name, "D")
            outdir_refl_cos = os.path.join(OUTPUT_BASE, year, month_name, "E")
        else:
            outdir_perez = os.path.join(OUTPUT_BASE, year, month_name)
            outdir_refl_cos = os.path.join(OUTPUT_BASE, year, month_name)

        print(f"\n>>> Memulai Pemrosesan Perez Periode [{idx}/{len(PERIODS)}]: {month_label} ({month_name.upper()}) <<<")
        process_perez_month(
            input_nc=input_nc,
            month_str=month_label,
            outdir_perez=outdir_perez,
            outdir_refl_cos=outdir_refl_cos,
        )

    print("\n" + "=" * 72)
    print("SELURUH PEMROSESAN 3 BULAN MODEL PEREZ SELESAI DENGAN SUKSES!")
    print("=" * 72)


if __name__ == "__main__":
    main()
