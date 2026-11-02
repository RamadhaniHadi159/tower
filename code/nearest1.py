import os
import sys
import xarray as xr
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

# Pilihan Musim yang Diproses (contoh: "son", "mam", "djf", "jja")
SEASON = "custom" # Pilihan: "djf", "mam", "jja", "son", atau "custom"

# Pengaturan Form Csat (Selaras dengan combined3_v6.py):
# Menjalankan Form D dan Form E sekaligus dalam satu kali eksekusi
# - "D" = Model Perez et al. (2002) standar
# - "E" = Model refl_cos
FORMS_TO_PROCESS = ["D", "E"]

# Template direktori file input GHI NetCDF
INPUT_GHI_DIR = (
    "/data/mahasiswa/romadhon/perez1/"
    "ghi/combined/nc"
)

# Direktori dasar output (dipertahankan sesuai versi sebelumnya)
OUTPUT_DIR = (
    "/data/mahasiswa/romadhon/perez1/"
    "ghi/bmkg"
)

# Konfigurasi stasiun BMKG Juanda & Karangploso
# Verifikasi koordinat resmi:
# Juanda: Stasiun Meteorologi Kelas I Juanda Sidoarjo (-7.3798, 112.7872)
# Karangploso: Stasiun Klimatologi Jawa Timur Karangploso Malang (-7.901117, 112.597553)
STATIONS = {
    "Juanda": {
        "name": "BMKG Juanda",
        "lat": -7.3798,
        "lon": 112.7872,
    },
    "Karangploso": {
        "name": "BMKG Karangploso",
        "lat": -7.901117,
        "lon": 112.597553,
    },
}

# Ambang batas maksimum jarak terdekat (km)
MAX_DISTANCE_KM = 30.0

# Hentikan eksekusi jika jarak stasiun melebihi ambang batas maksimum
STRICT_MAX_DISTANCE = True

# Toleransi komparasi floating-point
ATOL = 1e-6


# ============================================================
# FUNGSI BANTUAN
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2, radius_km=6371.0):
    """
    Menghitung jarak lingkaran besar (great-circle distance)
    antara dua titik koordinat dalam satuan kilometer (km)
    menggunakan formula Haversine.
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
    return radius_km * c


def detect_csat_form(ds, ghi_file_path):
    """
    Mendeteksi apakah data GHI merupakan formD (Perez) atau formE (refl_cos)
    berdasarkan isi data numerik (Csat vs formula), atribut NetCDF, atau pola path.
    Menggunakan np.isclose(atol=1e-6) untuk pengujian kesesuaian nilai Csat.
    """
    vars_available = set(ds.data_vars.keys()) | set(ds.coords.keys())
    
    # 1. Cek langsung dari nilai numerik data jika Csat & variabel pendukung ada
    needed = {"Csat", "reflectance", "sza_deg"}
    if needed.issubset(vars_available):
        try:
            sub_vars = ["Csat", "reflectance", "sza_deg"]
            if "am" in vars_available:
                sub_vars.append("am")
            if "epsilon" in vars_available:
                sub_vars.append("epsilon")

            sample_ds = ds[sub_vars]
            if "obs" in sample_ds.dims and sample_ds.dims["obs"] > 1000:
                sample_ds = sample_ds.isel(obs=slice(0, 1000))
            
            sample_df = sample_ds.to_dataframe().dropna()
            if len(sample_df) > 0:
                h = np.clip(90.0 - sample_df["sza_deg"], 1.0, 90.0)
                denom = 2.283 * (h ** (-0.26)) * np.exp(0.004 * h)
                c_raw = sample_df["reflectance"]
                am = sample_df["am"] if "am" in sample_df.columns else 1.0
                eps = sample_df["epsilon"] if "epsilon" in sample_df.columns else 1.0

                csat_d = (c_raw * am * eps) / denom
                csat_e = c_raw / denom

                match_d = np.isclose(sample_df["Csat"], csat_d, atol=ATOL).sum()
                match_e = np.isclose(sample_df["Csat"], csat_e, atol=ATOL).sum()

                if match_d > match_e and match_d > 0:
                    return "formD"
                elif match_e > match_d and match_e > 0:
                    return "formE"
        except Exception as err:
            print(f"Catatan: Verifikasi numerik Csat dilewati ({err}). Melanjutkan ke metadata/path.")

    # 2. Cek atribut global dataset
    for attr_key in ["form", "csat_form", "model_form", "title", "description"]:
        val = str(ds.attrs.get(attr_key, "")).lower()
        if "formd" in val or "perez" in val:
            return "formD"
        if "forme" in val or "refl_cos" in val:
            return "formE"

    # 3. Cek pola nama direktori / file input
    path_lower = ghi_file_path.lower()
    if "/d/" in path_lower or "formd" in path_lower or "_d." in path_lower:
        return "formD"
    if "/e/" in path_lower or "forme" in path_lower or "_e." in path_lower:
        return "formE"

    # Default fallback
    return "formD"


def resolve_output_path(template_path, form_tag):
    """
    Menyisipkan penanda form (formD atau formE) ke dalam nama file output.
    Contoh: 'nearest_son_bmkg_titik.txt' -> 'nearest_son_bmkg_titik_formD.txt'
    """
    directory = os.path.dirname(template_path)
    filename = os.path.basename(template_path)
    name, ext = os.path.splitext(filename)

    if "formD" in name or "formE" in name:
        return template_path
    
    new_filename = f"{name}_{form_tag}{ext}"
    return os.path.join(directory, new_filename)


def make_bounds(coord):
    """
    Membuat cell bounds 1D dari koordinat pusat grid 1D (selaras dengan regrid10).
    Digunakan untuk menentukan batas geometrik sel Masingar secara tepat.
    """
    coord = np.asarray(coord, dtype=np.float64)
    if coord.ndim != 1 or len(coord) < 2:
        raise ValueError(
            f"Koordinat grid harus 1D dan minimal 2 titik. "
            f"Ditemukan dimensi {coord.ndim} dan panjang {len(coord)}."
        )

    if not np.all(np.diff(coord) > 0):
        raise ValueError("Koordinat grid harus terurut monoton naik untuk pembentukan batas sel.")

    bounds = np.empty(len(coord) + 1, dtype=np.float64)
    bounds[1:-1] = (coord[:-1] + coord[1:]) / 2.0
    bounds[0] = coord[0] - (coord[1] - coord[0]) / 2.0
    bounds[-1] = coord[-1] + (coord[-1] - coord[-2]) / 2.0
    return bounds


def compute_bilinear_weights(st_lat, st_lon, unique_lats, unique_lons):
    """
    Menemukan 4 sel grid yang mengelilingi koordinat stasiun (SW, SE, NW, NE)
    dan menghitung bobot interpolasi bilinear berdasarkan jarak ke pusat tiap sel.
    Mengembalikan:
      - corner_weights: dictionary {(lat, lon): bobot}
      - bounds_info: tuple (lat_bottom, lat_top, lon_left, lon_right, u, v)
    """
    # Batas latitude (lat_bottom <= st_lat <= lat_top)
    lats_below = unique_lats[unique_lats <= st_lat]
    lats_above = unique_lats[unique_lats >= st_lat]

    lat_bottom = lats_below.max() if len(lats_below) > 0 else unique_lats[0]
    lat_top = lats_above.min() if len(lats_above) > 0 else unique_lats[-1]

    # Batas longitude (lon_left <= st_lon <= lon_right)
    lons_left = unique_lons[unique_lons <= st_lon]
    lons_right = unique_lons[unique_lons >= st_lon]

    lon_left = lons_left.max() if len(lons_left) > 0 else unique_lons[0]
    lon_right = lons_right.min() if len(lons_right) > 0 else unique_lons[-1]

    # Normalisasi koordinat lokal [0, 1]
    if lat_top > lat_bottom:
        v = (st_lat - lat_bottom) / (lat_top - lat_bottom)
    else:
        v = 0.0

    if lon_right > lon_left:
        u = (st_lon - lon_left) / (lon_right - lon_left)
    else:
        u = 0.0

    # Bobot untuk masing-masing sudut:
    # SW: (lat_bottom, lon_left)
    # SE: (lat_bottom, lon_right)
    # NW: (lat_top, lon_left)
    # NE: (lat_top, lon_right)
    raw_corners = [
        (lat_bottom, lon_left, (1.0 - u) * (1.0 - v), "SW"),
        (lat_bottom, lon_right, u * (1.0 - v), "SE"),
        (lat_top, lon_left, (1.0 - u) * v, "NW"),
        (lat_top, lon_right, u * v, "NE"),
    ]

    corner_weights = {}
    for clat, clon, cw, cname in raw_corners:
        key = (float(clat), float(clon))
        corner_weights[key] = corner_weights.get(key, 0.0) + float(cw)

    return corner_weights, (lat_bottom, lat_top, lon_left, lon_right, u, v)


# ============================================================
# PEMROSESAN PER FORM (1 SEL & 1 TITIK STASIUN BMKG)
# ============================================================

def process_nearest_for_form(form_tag, season_name):
    form_label = f"form{form_tag.upper()}"
    form_desc = "Perez et al. (2002)" if form_tag.upper() == "D" else "refl_cos"

    print("\n" + "=" * 76)
    print(f"PEMROSESAN NEAREST BMKG — {form_label.upper()} ({form_desc}) — MUSIM {season_name.upper()}")
    print("=" * 76)

    # 1. Tentukan file input GHI NetCDF dari combined3_v6.py
    ghi_file = os.path.join(
        INPUT_GHI_DIR,
        f"ghi_{season_name.lower()}_season_{form_label}.nc"
    )

    # Fallback jika file spesifik form tidak ditemukan
    if not os.path.isfile(ghi_file):
        fallback_file = os.path.join(
            INPUT_GHI_DIR,
            f"ghi_{season_name.lower()}_season.nc"
        )
        if os.path.isfile(fallback_file):
            print(f"Pemberitahuan: File spesifik form tidak ditemukan, beralih ke fallback: {fallback_file}")
            ghi_file = fallback_file

    print(f"File Input NetCDF   : {ghi_file}")
    if not os.path.isfile(ghi_file):
        raise FileNotFoundError(
            f"File input NetCDF tidak ditemukan: {ghi_file}\n"
            f"Pastikan combined3_v6.py telah dieksekusi untuk musim {season_name.upper()} ({form_label})."
        )

    # 2. Buka dataset dan deteksi form
    ds = xr.open_dataset(ghi_file)
    detected_form = detect_csat_form(ds, ghi_file)

    # 3. Tentukan nama file output
    output_titik_template = os.path.join(OUTPUT_DIR, f"nearest_{season_name.lower()}_bmkg_titik.txt")
    output_set_template = os.path.join(OUTPUT_DIR, f"nearest_{season_name.lower()}_bmkg_set.txt")

    output_titik = resolve_output_path(output_titik_template, detected_form)
    output_set = resolve_output_path(output_set_template, detected_form)

    print(f"Form Terdeteksi     : {detected_form}")
    print(f"Output Titik Target : {output_titik} (Metode 1 Titik: Bilinear Interpolation)")
    print(f"Output Set Target   : {output_set} (Metode 1 Sel: Nearest Box Masingar)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 4. Validasi variabel wajib (termasuk qc_TL)
    required_vars = [
        "valid_time",
        "latitude",
        "longitude",
        "ssrd_Wm2",
        "GHI",
        "qc_TL",
    ]

    available_vars = set(ds.data_vars.keys()) | set(ds.coords.keys())
    missing_vars = [v for v in required_vars if v not in available_vars]
    if missing_vars:
        raise KeyError(f"Variabel wajib berikut tidak ditemukan dalam dataset: {missing_vars}")

    df = (
        ds[required_vars]
        .to_dataframe()
        .reset_index()
    )
    ds.close()

    print(f"\nData berhasil dibaca: {len(df):,} observasi")

    # 5. Penyelarasan kolom dan tipe data numerik
    df = df.rename(columns={"ssrd_Wm2": "ssrd(W/m²)"})
    df["valid_time"] = pd.to_datetime(df["valid_time"], errors="coerce")
    df["ssrd(W/m²)"] = pd.to_numeric(df["ssrd(W/m²)"], errors="coerce")
    df["GHI"] = pd.to_numeric(df["GHI"], errors="coerce")
    df["qc_TL"] = pd.to_numeric(df["qc_TL"], errors="coerce")

    # 6. Ambil daftar koordinat kisi unik pada domain dan hitung batas sel Masingar
    grid = (
        df[["latitude", "longitude"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    unique_lats = np.sort(grid["latitude"].unique())
    unique_lons = np.sort(grid["longitude"].unique())

    # Pembuatan batas sel Masingar 1D (make_bounds)
    lat_bounds = make_bounds(unique_lats)
    lon_bounds = make_bounds(unique_lons)

    print(f"Dimensi Grid Domain : {len(unique_lats)} lintang x {len(unique_lons)} bujur = {len(grid)} total sel")

    station_summaries = []
    station_timeseries_set = []
    station_timeseries_titik = []

    print("\n" + "-" * 76)
    print(f"ANALISIS SPASIAL: 1 SEL & 1 TITIK BMKG ({detected_form})")
    print("-" * 76)

    for st_key, st_info in STATIONS.items():
        st_name = st_info.get("name", st_key)
        st_lat = st_info.get("lat")
        st_lon = st_info.get("lon")

        # Validasi jika konfigurasi koordinat belum lengkap
        if st_lat is None or st_lon is None or np.isnan(st_lat) or np.isnan(st_lon):
            raise ValueError(
                f"Koordinat untuk stasiun '{st_name}' ({st_key}) belum tersedia (lat/lon bernilai None/NaN)! "
                f"Harap lengkapi konfigurasi koordinat stasiun sebelum memproses."
            )

        # ----------------------------------------------------
        # BAGIAN A: 1 SEL (Pusat Sel Kisi Masingar Terdekat)
        # ----------------------------------------------------
        distances_km = haversine_km(
            st_lat,
            st_lon,
            grid["latitude"],
            grid["longitude"]
        )

        nearest_idx = int(np.argmin(distances_km.values))
        nearest_grid = grid.iloc[nearest_idx]
        nearest_lat = float(nearest_grid["latitude"])
        nearest_lon = float(nearest_grid["longitude"])
        nearest_dist = float(distances_km.iloc[nearest_idx])

        # Verifikasi batas toleransi jarak maksimum (km)
        status_dist = "VALID" if nearest_dist <= MAX_DISTANCE_KM else "EXCEEDED"

        # Verifikasi posisi stasiun menggunakan batas sel resmi grid Masingar:
        idx_lat = int(np.where(np.isclose(unique_lats, nearest_lat, atol=ATOL))[0][0])
        idx_lon = int(np.where(np.isclose(unique_lons, nearest_lon, atol=ATOL))[0][0])

        lat_cell_min = float(lat_bounds[idx_lat])
        lat_cell_max = float(lat_bounds[idx_lat + 1])
        lon_cell_min = float(lon_bounds[idx_lon])
        lon_cell_max = float(lon_bounds[idx_lon + 1])

        # Stasiun berada di dalam sel jika posisinya berada di antara batas bawah dan batas atas sel
        contains_station = (
            (lat_cell_min <= st_lat <= lat_cell_max)
            and (lon_cell_min <= st_lon <= lon_cell_max)
        )
        status_containment = "TERPENUHI (Stasiun di dalam batas sel Masingar)" if contains_station else "DI LUAR BATAS SEL MASINGAR"

        dlat = abs(st_lat - nearest_lat)
        dlon = abs(st_lon - nearest_lon)

        print(f"\n========================================================")
        print(f"Stasiun: {st_name} ({st_key})")
        print(f"========================================================")
        print(f"1. ANALISIS METODE 1 SEL (Pusat Sel Kisi Masingar):")
        print(f"   - Koordinat Stasiun In-Situ : {st_lat:.6f}, {st_lon:.6f}")
        print(f"   - Pusat Sel Masingar        : {nearest_lat:.6f}, {nearest_lon:.6f}")
        print(f"   - Jarak ke Pusat Sel        : {nearest_dist:.3f} km (Batas: {MAX_DISTANCE_KM:.1f} km, Status: {status_dist})")
        print(f"   - Batas Lintang Sel Masingar: [{lat_cell_min:.6f}°, {lat_cell_max:.6f}°] (Rentang: {lat_cell_max - lat_cell_min:.4f}°)")
        print(f"   - Batas Bujur Sel Masingar  : [{lon_cell_min:.6f}°, {lon_cell_max:.6f}°] (Rentang: {lon_cell_max - lon_cell_min:.4f}°)")
        print(f"   - Selisih |Δlat| & |Δlon|   : |Δlat|={dlat:.4f}°, |Δlon|={dlon:.4f}°")
        print(f"   - Validasi Muatan Sel       : {status_containment}")

        # Hentikan proses jika jarak > batas maksimum
        if nearest_dist > MAX_DISTANCE_KM and STRICT_MAX_DISTANCE:
            raise ValueError(
                f"Jarak grid terdekat stasiun {st_name} ({nearest_dist:.3f} km) "
                f"melebihi batas toleransi maksimum {MAX_DISTANCE_KM:.1f} km! Proses dihentikan."
            )

        # Ambil data deret waktu pada 1 sel terdekat apa adanya
        mask_nearest = (
            np.isclose(df["latitude"], nearest_lat, atol=ATOL)
            &
            np.isclose(df["longitude"], nearest_lon, atol=ATOL)
        )
        df_st_raw = df[mask_nearest].copy()

        if df_st_raw.empty:
            raise ValueError(
                f"Tidak ada data yang ditemukan untuk stasiun {st_name} pada sel "
                f"({nearest_lat}, {nearest_lon}) dengan toleransi atol={ATOL}."
            )

        # Urutkan berdasarkan waktu dan periksa duplikat waktu (raise ValueError, jangan hapus diam-diam)
        df_st_raw = df_st_raw.sort_values("valid_time").reset_index(drop=True)
        dup_mask = df_st_raw.duplicated(subset=["valid_time"], keep=False)
        n_duplicates = int(dup_mask.sum())
        if n_duplicates > 0:
            sample_dups = df_st_raw.loc[dup_mask, "valid_time"].head(10).tolist()
            raise ValueError(
                f"Ditemukan {n_duplicates} baris waktu duplikat pada stasiun {st_name} (sel {nearest_lat}, {nearest_lon})! "
                f"Contoh waktu duplikat: {sample_dups}. Proses dihentikan."
            )

        # Filter NaN untuk data numerik wajib (mempertahankan qc_TL)
        valid_mask = (
            df_st_raw["valid_time"].notna()
            & df_st_raw["ssrd(W/m²)"].notna()
            & df_st_raw["GHI"].notna()
            & df_st_raw["qc_TL"].notna()
        )
        df_st_valid = df_st_raw[valid_mask].copy().reset_index(drop=True)
        df_st_valid["qc_TL"] = df_st_valid["qc_TL"].astype(int)

        print(f"   - Observasi Valid 1 Sel     : {len(df_st_valid):,} data jam-jaman")
        print(f"   - Distribusi qc_TL 1 Sel    : {df_st_valid['qc_TL'].value_counts().to_dict()} (0=LOLOS, 1=TL>10)")

        # Bentuk DataFrame 1 SEL (sertakan kolom qc_TL)
        df_out_set = pd.DataFrame({
            "datetime": df_st_valid["valid_time"],
            "station": st_key,
            "latitude": nearest_lat,
            "longitude": nearest_lon,
            "ssrd(W/m²)": df_st_valid["ssrd(W/m²)"],
            "GHI": df_st_valid["GHI"],
            "qc_TL": df_st_valid["qc_TL"],
        })
        station_timeseries_set.append(df_out_set)

        # ----------------------------------------------------
        # BAGIAN B: 1 TITIK (Interpolasi Bilinear 4 Sel)
        # ----------------------------------------------------
        corner_weights, bounds_info = compute_bilinear_weights(
            st_lat, st_lon, unique_lats, unique_lons
        )
        lat_b, lat_t, lon_l, lon_r, u_norm, v_norm = bounds_info

        print(f"\n2. ANALISIS METODE 1 TITIK (Bilinear Interpolation 4 Sel):")
        print(f"   - Koordinat Titik Target    : {st_lat:.6f}, {st_lon:.6f}")
        print(f"   - Lintang Pengapit          : [{lat_b:.6f}, {lat_t:.6f}] -> v = {v_norm:.4f}")
        print(f"   - Bujur Pengapit            : [{lon_l:.6f}, {lon_r:.6f}] -> u = {u_norm:.4f}")
        print(f"   - Bobot Tetap 4 Sel Sekitar :")

        sorted_corners = sorted(corner_weights.items(), key=lambda x: x[1], reverse=True)
        for (clat, clon), cw in sorted_corners:
            cdist = haversine_km(st_lat, st_lon, clat, clon)
            is_nearest_mark = " (<- Sel Terdekat)" if np.isclose(clat, nearest_lat, atol=ATOL) and np.isclose(clon, nearest_lon, atol=ATOL) else ""
            print(f"     * Sel ({clat:.6f}, {clon:.6f}) : Bobot = {cw * 100:.2f}% | Jarak = {cdist:.3f} km{is_nearest_mark}")

        # Siapkan data dari keempat sel pengeliling dan periksa duplikat waktu secara ketat
        corner_series_dict = {}
        for (clat, clon), cw in corner_weights.items():
            if cw <= 0.0:
                continue

            mask_c = (
                np.isclose(df["latitude"], clat, atol=ATOL)
                & np.isclose(df["longitude"], clon, atol=ATOL)
            )
            df_c = df.loc[mask_c].sort_values("valid_time").reset_index(drop=True)

            # Periksa duplikat waktu pada sel pengeliling (raise ValueError jika ada)
            dup_c = df_c.duplicated(subset=["valid_time"], keep=False)
            if dup_c.sum() > 0:
                sample_dups = df_c.loc[dup_c, "valid_time"].head(10).tolist()
                raise ValueError(
                    f"Ditemukan {dup_c.sum()} baris waktu duplikat pada sel pengeliling ({clat}, {clon})! "
                    f"Contoh waktu duplikat: {sample_dups}. Proses dihentikan."
                )

            # Identifikasi baris valid pada sel ini
            valid_c = (
                df_c["valid_time"].notna()
                & df_c["ssrd(W/m²)"].notna()
                & df_c["GHI"].notna()
                & df_c["qc_TL"].notna()
            )
            df_c_clean = df_c[valid_c].set_index("valid_time")
            corner_series_dict[(clat, clon)] = {
                "weight": cw,
                "df": df_c_clean,
            }

        # Pemilihan waktu: HANYA waktu di mana KEEMPAT sel yang diperlukan valid bersamaan
        common_times = None
        for (clat, clon), cdata in corner_series_dict.items():
            times_this_corner = set(cdata["df"].index)
            if common_times is None:
                common_times = times_this_corner
            else:
                common_times = common_times.intersection(times_this_corner)

        if not common_times:
            raise ValueError(
                f"Tidak ditemukan waktu bersamaan di mana keempat sel pengeliling valid untuk stasiun {st_name}!"
            )

        sorted_common_times = pd.DatetimeIndex(sorted(list(common_times)))

        # Interpolasi bilinear tanpa normalisasi ulang bobot (karena keempat sel 100% valid)
        interp_ssrd = pd.Series(0.0, index=sorted_common_times)
        interp_ghi = pd.Series(0.0, index=sorted_common_times)
        qc_tl_flags = []

        for (clat, clon), cdata in corner_series_dict.items():
            cw = cdata["weight"]
            df_corner = cdata["df"].reindex(sorted_common_times)

            interp_ssrd += cw * df_corner["ssrd(W/m²)"]
            interp_ghi += cw * df_corner["GHI"]
            qc_tl_flags.append(df_corner["qc_TL"].astype(int))

        # QC konsisten untuk 1 titik: Jika ada sel pengeliling dengan qc_TL = 1 (TL > 10), titik ditandai qc_TL = 1
        qc_tl_titik = np.maximum.reduce(qc_tl_flags)

        # Bentuk DataFrame 1 TITIK (sertakan kolom qc_TL)
        df_out_titik = pd.DataFrame({
            "datetime": sorted_common_times,
            "station": st_key,
            "latitude": float(st_lat),
            "longitude": float(st_lon),
            "ssrd(W/m²)": interp_ssrd.values,
            "GHI": interp_ghi.values,
            "qc_TL": qc_tl_titik,
        })
        station_timeseries_titik.append(df_out_titik)

        print(f"   - Observasi Valid 1 Titik   : {len(df_out_titik):,} data jam-jaman (keempat sel valid serentak)")
        print(f"   - Distribusi qc_TL 1 Titik  : {df_out_titik['qc_TL'].value_counts().to_dict()} (0=LOLOS, 1=TL>10)")

        # Simpan ringkasan audit spasial
        station_summaries.append({
            "station": st_key,
            "station_name": st_name,
            "target_lat": st_lat,
            "target_lon": st_lon,
            "nearest_lat": nearest_lat,
            "nearest_lon": nearest_lon,
            "distance_km": round(nearest_dist, 3),
            "dlat_deg": round(dlat, 4),
            "dlon_deg": round(dlon, 4),
            "contains_station": "YA" if contains_station else "TIDAK",
            "lat_cell_range": f"[{lat_cell_min:.3f}, {lat_cell_max:.3f}]",
            "lon_cell_range": f"[{lon_cell_min:.3f}, {lon_cell_max:.3f}]",
            "status": status_dist,
        })

    # 7. Penyusunan Output
    df_summary = pd.DataFrame(station_summaries)

    # Gabungan deret waktu untuk TITIK (menerapkan interpolasi bilinear pada koordinat stasiun)
    df_titik = pd.concat(station_timeseries_titik, ignore_index=True)
    df_titik = df_titik.sort_values(by=["station", "datetime"]).reset_index(drop=True)

    # Gabungan deret waktu untuk SET (menerapkan 1 sel terdekat apa adanya pada pusat sel)
    df_set = pd.concat(station_timeseries_set, ignore_index=True)
    df_set = df_set.sort_values(by=["station", "datetime"]).reset_index(drop=True)

    print("\n" + "=" * 76)
    print("RINGKASAN AUDIT SPASIAL & BATAS SEL MASINGAR")
    print("=" * 76)
    print(df_summary.to_string(index=False))

    print("\n" + "-" * 76)
    print(f"PREVIEW DERET WAKTU 1 TITIK DENGAN qc_TL ({os.path.basename(output_titik)})")
    print("-" * 76)
    print(df_titik.head(10).to_string(index=False))
    print(f"\nTotal baris deret waktu 1 titik : {len(df_titik):,}")

    print("\n" + "-" * 76)
    print(f"PREVIEW GABUNGAN DERET WAKTU 1 SEL DENGAN qc_TL ({os.path.basename(output_set)})")
    print("-" * 76)
    print(df_set.head(10).to_string(index=False))
    print(f"\nTotal baris gabungan deret waktu 1 sel : {len(df_set):,}")

    # 8. Simpan Output ke File (Masing-masing 7 kolom: datetime, station, lat, lon, ssrd, GHI, qc_TL)
    # File Titik: deret waktu lengkap dengan koordinat titik stasiun BMKG (presisi 6 desimal)
    if os.path.exists(output_titik):
        os.remove(output_titik)
    df_titik.to_csv(
        output_titik,
        sep="\t",
        index=False,
        float_format="%.6f"
    )
    print(f"\nFile 1 titik berhasil disimpan : {output_titik}")

    # File Set: deret waktu lengkap 1 sel terdekat apa adanya (presisi 3 desimal)
    if os.path.exists(output_set):
        os.remove(output_set)
    df_set.to_csv(
        output_set,
        sep="\t",
        index=False,
        float_format="%.3f"
    )
    print(f"File 1 sel berhasil disimpan   : {output_set}")


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main():
    print("=" * 76)
    print("PROGRAM EKSTRAKSI 1 SEL & 1 TITIK BMKG (NEAREST V8)")
    print("=" * 76)
    print(f"Musim Diproses     : {SEASON.upper()}")
    print(f"Form Diproses      : {FORMS_TO_PROCESS}")
    print(f"Input NetCDF       : {INPUT_GHI_DIR}")
    print(f"Direktori Output   : {OUTPUT_DIR}")
    print("=" * 76)

    for form in FORMS_TO_PROCESS:
        process_nearest_for_form(
            form_tag=form,
            season_name=SEASON,
        )

    print("\n" + "=" * 76)
    print("SELURUH PROSES EKSTRAKSI 1 SEL & 1 TITIK BERHASIL DISELESAIKAN!")
    print("=" * 76)


if __name__ == "__main__":
    main()
