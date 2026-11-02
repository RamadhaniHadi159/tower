import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy.interpolate import griddata

# Kompatibilitas Shapely 2.0+ dan Shapely 1.8.x
try:
    from shapely import contains_xy
except ImportError:
    try:
        from shapely.vectorized import contains as contains_xy
    except ImportError:
        def contains_xy(geom, x, y):
            from shapely.geometry import Point
            pts = [Point(xi, yi) for xi, yi in zip(np.asarray(x).ravel(), np.asarray(y).ravel())]
            res = np.array([geom.contains(p) for p in pts], dtype=bool)
            return res.reshape(np.asarray(x).shape)

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FormatStrFormatter


# ============================================================
# 1. KONFIGURASI UMUM & TAMPILAN
# ============================================================

plt.rcParams["savefig.dpi"] = 1000
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# Pilihan Colormap:
# Menggunakan truncated YlOrRd (dimulai dari 0.12 s.d. 1.00) agar nilai terendah
# (JJA/MAM) tidak berwarna kuning keputihan yang pucat menyatu dengan latar putih,
# melainkan kuning keemasan solid yang kontras dan jelas isolinenya.
base_cmap = plt.get_cmap("YlOrRd")
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "YlOrRd_solid",
    base_cmap(np.linspace(0.12, 1.0, 256))
)

N_INTERP = 500
N_TICKS = 7


# ============================================================
# 2. KONFIGURASI INPUT (SELARAS DENGAN combined3_v6.py)
# ============================================================

# Jalur input utama NetCDF hasil penggabungan dari combined3_v6.py
INPUT_GHI_DIR = "/data/mahasiswa/romadhon/perez1/ghi/combined/nc"

# Fallback otomatis jika folder di atas berada langsung di combined/nc
if not os.path.exists(INPUT_GHI_DIR):
    alt_dir = "/data/mahasiswa/romadhon/perez1/ghi/combined/nc"
    if os.path.exists(alt_dir):
        INPUT_GHI_DIR = alt_dir

# Pilihan Form model Csat yang diproses:
# - "D" : Perez et al. (2002) standar
# - "E" : refl_cos
# Bisa diproses satu form (misal ["D"]) atau keduanya (["D", "E"])
FORMS_TO_PROCESS = ["D"]

# Daftar musim yang akan ditampilkan dalam 4 panel (2x2)
SEASONS = [
    ("DJF", "DJF Season"),
    ("MAM", "MAM Season"),
    ("JJA", "JJA Season"),
    ("SON", "SON Season")
]


# ============================================================
# 3. FUNGSI RESOLVER FILE INPUT COMBINED
# ============================================================

def resolve_season_nc(input_dir, season_code, form_tag):
    """
    Mencari file NetCDF musiman luaran resmi dari combined3_v6.py.
    Format utama: ghi_<season>_season_form<tag>.nc
    Contoh: ghi_djf_season_formD.nc
    """
    season_lower = season_code.lower()
    form_label = f"form{form_tag.upper()}"

    candidates = [
        # 1. Output resmi combined3_v6.py dengan label form:
        os.path.join(input_dir, f"ghi_{season_lower}_season_{form_label}.nc"),
        # 2. Output tanpa label form:
        os.path.join(input_dir, f"ghi_{season_lower}_season.nc"),
        # 3. Output dengan huruf kapital:
        os.path.join(input_dir, f"ghi_{season_code.upper()}_season_{form_label}.nc"),
        os.path.join(input_dir, f"ghi_{season_code.upper()}_season.nc"),
    ]

    for cand in candidates:
        if os.path.isfile(cand):
            return cand

    # Default ke kandidat pertama jika belum tersedia di disk lokal
    return candidates[0]


# ============================================================
# 4. SHAPEFILE JAWA TIMUR & SEKITARNYA
# ============================================================

shapefile = (
    "/data/mahasiswa/romadhon/perez1/gdam/"
    "gadm41_IDN_2.shp"
)

print("Membaca shapefile...")

# Cek keberadaan shapefile dengan fallback folder gadm/gdam
if not os.path.exists(shapefile):
    alt_shapefile = shapefile.replace("/gdam/", "/gadm/")
    if os.path.exists(alt_shapefile):
        print(f"Menggunakan jalur alternatif shapefile: {alt_shapefile}")
        shapefile = alt_shapefile
    else:
        raise FileNotFoundError(
            f"Shapefile tidak ditemukan:\n{shapefile}"
        )

gdf = gpd.read_file(
    shapefile
).to_crs(epsg=4326)

jateng = gdf[gdf["NAME_1"] == "Jawa Tengah"].copy()
jatim = gdf[gdf["NAME_1"] == "Jawa Timur"].copy()
bali = gdf[gdf["NAME_1"] == "Bali"].copy()

if jateng.empty or jatim.empty or bali.empty:
    raise ValueError("Wilayah Jawa Tengah, Jawa Timur, atau Bali tidak lengkap di shapefile.")

# Geometri gabungan Jawa Timur
if hasattr(jatim.geometry, "union_all"):
    jatim_union = jatim.geometry.union_all()
else:
    jatim_union = jatim.geometry.unary_union

# Background wilayah pendukung
background = gpd.GeoDataFrame(
    pd.concat([jateng, jatim, bali], ignore_index=True),
    crs=gdf.crs
)


# ============================================================
# 5. BATAS KOORDINAT PETA (DISESUAIKAN)
# ============================================================

map_minx = 110.8
map_maxx = 116.0

map_miny = -9.0
map_maxy = -5.7


# ============================================================
# 6. FUNGSI PROSES DATA SATU MUSIM (DENGAN INTERPOLASI 3 TAHAP)
# ============================================================

def proses_season_file(file_path, nama_season):
    print("\n" + "=" * 60)
    print(f"MEMPROSES {nama_season}")
    print(f"File Input : {file_path}")
    print("=" * 60)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"File input NetCDF tidak ditemukan: {file_path}\n"
            f"Pastikan combined3_v6.py telah dieksekusi untuk menghasilkan file musiman ini."
        )

    ghi_candidates = ["GHI", "ghi", "GHI_Wm2", "ghi_Wm2"]
    lat_candidates = ["latitude", "lat", "LATITUDE", "LAT"]
    lon_candidates = ["longitude", "lon", "LONGITUDE", "LON"]

    with xr.open_dataset(file_path) as ds:
        df = ds.to_dataframe().reset_index()

        ghi_col = next((c for c in ghi_candidates if c in df.columns), None)
        if ghi_col is None:
            raise ValueError(
                f"Kolom GHI tidak ditemukan pada {file_path}.\n"
                f"Kolom tersedia: {df.columns.tolist()}"
            )

        lat_col = next((c for c in lat_candidates if c in df.columns), None)
        if lat_col is None:
            raise ValueError(
                f"Kolom Latitude tidak ditemukan pada {file_path}.\n"
                f"Kolom tersedia: {df.columns.tolist()}"
            )

        lon_col = next((c for c in lon_candidates if c in df.columns), None)
        if lon_col is None:
            raise ValueError(
                f"Kolom Longitude tidak ditemukan pada {file_path}.\n"
                f"Kolom tersedia: {df.columns.tolist()}"
            )

        # Hanya ekstrak subset kolom yang relevan untuk efisiensi RAM
        sub_df = pd.DataFrame({
            "GHI": pd.to_numeric(df[ghi_col], errors="coerce"),
            "latitude": pd.to_numeric(df[lat_col], errors="coerce").round(5),
            "longitude": pd.to_numeric(df[lon_col], errors="coerce").round(5)
        })

    # Cleaning data
    sub_df = sub_df.dropna(subset=["GHI", "latitude", "longitude"])
    sub_df = sub_df[sub_df["GHI"] >= 0]

    # Rata-rata GHI musiman per grid
    rata = sub_df.groupby(["latitude", "longitude"], as_index=False)["GHI"].mean()

    print(f"Jumlah grid : {len(rata):,}")
    print(f"Minimum     : {rata['GHI'].min():.2f} W/m²")
    print(f"Maksimum    : {rata['GHI'].max():.2f} W/m²")

    x = rata["longitude"].values
    y = rata["latitude"].values
    z = rata["GHI"].values

    grid_lon = np.linspace(map_minx, map_maxx, N_INTERP)
    grid_lat = np.linspace(map_miny, map_maxy, N_INTERP)
    grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lon, grid_lat)

    # 1. Interpolasi cubic utama untuk kelancaran kontur
    print(f"Interpolasi cubic : {nama_season}")
    grid_ghi = griddata(
        points=(x, y),
        values=z,
        xi=(grid_lon_2d, grid_lat_2d),
        method="cubic"
    )

    # 2. Fallback interpolasi linear untuk batas convex hull
    nan_mask = np.isnan(grid_ghi)
    if np.any(nan_mask):
        grid_linear = griddata(
            points=(x, y),
            values=z,
            xi=(grid_lon_2d, grid_lat_2d),
            method="linear"
        )
        grid_ghi[nan_mask] = grid_linear[nan_mask]

    # 3. Fallback interpolasi nearest neighbor untuk memastikan tepi pantai
    # dan pulau terluar tertutup sempurna tanpa lubang NaN sebelum masking
    nan_mask2 = np.isnan(grid_ghi)
    if np.any(nan_mask2):
        grid_nearest = griddata(
            points=(x, y),
            values=z,
            xi=(grid_lon_2d, grid_lat_2d),
            method="nearest"
        )
        grid_ghi[nan_mask2] = grid_nearest[nan_mask2]

    # Batas fisik: GHI tidak boleh negatif
    grid_ghi = np.clip(grid_ghi, a_min=0, a_max=None)

    # Masking poligon Jawa Timur
    try:
        mask_jatim = contains_xy(jatim_union, grid_lon_2d, grid_lat_2d)
    except Exception:
        mask_flat = contains_xy(jatim_union, grid_lon_2d.ravel(), grid_lat_2d.ravel())
        mask_jatim = np.asarray(mask_flat).reshape(grid_lon_2d.shape)

    grid_ghi[~mask_jatim] = np.nan

    return {
        "rata": rata,
        "grid_ghi": grid_ghi,
        "grid_lon_2d": grid_lon_2d,
        "grid_lat_2d": grid_lat_2d
    }


# ============================================================
# 7. PROSES PEMETAAN PER FORM
# ============================================================

output_dir = "/data/mahasiswa/romadhon/perez1/peta"
os.makedirs(output_dir, exist_ok=True)

for form_tag in FORMS_TO_PROCESS:
    form_label = f"form{form_tag.upper()}"
    form_desc = "Perez et al. (2002)" if form_tag.upper() == "D" else "refl_cos"

    print("\n" + "=" * 70)
    print(f"MEMBUAT PETA 4 MUSIM — {form_label.upper()} ({form_desc})")
    print(f"Sumber Input: {INPUT_GHI_DIR}")
    print("=" * 70)

    # Siapkan file input per musim dari combined3_v6.py
    season_file_map = {}
    for season_code, display_name in SEASONS:
        nc_path = resolve_season_nc(INPUT_GHI_DIR, season_code, form_tag)
        season_file_map[display_name] = nc_path
        print(f"Musim {season_code} -> {nc_path}")

    # Proses komputasi 4 musim
    hasil = {}
    for _, display_name in SEASONS:
        file_path = season_file_map[display_name]
        hasil[display_name] = proses_season_file(file_path, display_name)

    # Hitung rentang nilai gabungan untuk 1 colorbar terpadu
    semua_nilai = []
    for display_name in hasil:
        val = hasil[display_name]["rata"]["GHI"].values
        val = val[np.isfinite(val)]
        semua_nilai.append(val)

    semua_nilai = np.concatenate(semua_nilai)
    vmin = float(np.min(semua_nilai))
    vmax = float(np.max(semua_nilai))

    print("\n" + "=" * 60)
    print(f"SKALA COLORBAR ({form_label.upper()})")
    print("=" * 60)
    print(f"Minimum : {vmin:.2f} W/m²")
    print(f"Maksimum : {vmax:.2f} W/m²")

    norm = Normalize(vmin=vmin, vmax=vmax)
    contour_levels = np.linspace(vmin, vmax, 200)

    # Inisialisasi Figure
    fig = plt.figure(figsize=(14, 10))

    # Dimensi & Penempatan 4 Axes
    W = 0.385
    H = 0.350
    GAP_X = 0.045
    GAP_Y = 0.045

    X1 = 0.055
    X2 = X1 + W + GAP_X

    Y_BOTTOM = 0.105
    Y_TOP = Y_BOTTOM + H + GAP_Y

    ax1 = fig.add_axes([X1, Y_TOP, W, H])     # DJF (Kiri Atas)
    ax2 = fig.add_axes([X2, Y_TOP, W, H])     # MAM (Kanan Atas)
    ax3 = fig.add_axes([X1, Y_BOTTOM, W, H])  # JJA (Kiri Bawah)
    ax4 = fig.add_axes([X2, Y_BOTTOM, W, H])  # SON (Kanan Bawah)

    axes = [ax1, ax2, ax3, ax4]

    for ax, (_, display_name) in zip(axes, SEASONS):
        data = hasil[display_name]
        grid_lon_2d = data["grid_lon_2d"]
        grid_lat_2d = data["grid_lat_2d"]
        grid_ghi = data["grid_ghi"]

        # Kunci nilai hasil interpolasi tepat di rentang [vmin, vmax]
        # untuk memastikan tidak ada lubang akibat spline undershoot/overshoot
        grid_ghi_clamped = np.clip(grid_ghi, a_min=vmin, a_max=vmax)

        # Background wilayah
        background.plot(
            ax=ax,
            facecolor="white",
            edgecolor="black",
            linewidth=0.40,
            zorder=1
        )

        # Kontur GHI dengan data yang sudah dikunci ke rentang [vmin, vmax]
        ax.contourf(
            grid_lon_2d,
            grid_lat_2d,
            grid_ghi_clamped,
            levels=contour_levels,
            cmap=CMAP,
            norm=norm,
            extend="both",
            antialiased=True,
            zorder=3
        )

        # Batas administrasi Jawa Timur
        jatim.boundary.plot(
            ax=ax,
            color="black",
            linewidth=0.45,
            zorder=5
        )

        ax.set_xlim(map_minx, map_maxx)
        ax.set_ylim(map_miny, map_maxy)

        ax.set_xticks(np.arange(111, 117, 1))
        ax.set_yticks(np.arange(-9, -5, 1))

        ax.set_xlabel("Longitude (°)", fontsize=9, labelpad=2)
        ax.set_ylabel("Latitude (°)", fontsize=9, labelpad=2)
        ax.tick_params(axis="both", labelsize=8, pad=2)
        ax.set_title(display_name, fontsize=12, pad=4)
        ax.grid(False)

    # Colorbar terpadu di sebelah kanan:
    # Menggunakan extend="neither" agar ujung atas dan bawah colorbar berbentuk DATAR (persegi panjang),
    # bukan lancip (segitiga).
    sm = ScalarMappable(norm=norm, cmap=CMAP)
    sm.set_array([])

    cbar_ax = fig.add_axes([0.900, 0.135, 0.020, 0.680])
    cbar = fig.colorbar(
        sm,
        cax=cbar_ax,
        ticks=np.linspace(vmin, vmax, N_TICKS),
        extend="neither"
    )
    cbar.set_label("GHI (W m$^{-2}$)", fontsize=11, labelpad=10)
    cbar.ax.tick_params(labelsize=9, length=3, width=0.8)
    cbar.minorticks_off()
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    # Judul Utama
    fig.suptitle(
        f"Global Horizontal Irradiance ({form_label.upper()})",
        fontsize=18,
        y=0.900
    )

    # Simpan Output PDF versi 7
    output_filename = f"peta_ghi_jatim_seasons_{form_label}.pdf"
    output_filepath = os.path.join(output_dir, output_filename)

    plt.savefig(output_filepath, format="pdf", bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 60)
    print(f"PEMBUATAN PETA SELESAI ({form_label.upper()})")
    print(f"File Output : {output_filepath}")
    print("=" * 60)

print("\nSeluruh proses pemetaan selesai.")
