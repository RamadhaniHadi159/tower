import os
import sys
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
# pada bulan-bulan beradiasi rendah tidak berwarna kuning keputihan yang pucat
# menyatu dengan latar belakang, melainkan kuning keemasan solid yang kontras.
base_cmap = plt.get_cmap("YlOrRd")
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "YlOrRd_solid",
    base_cmap(np.linspace(0.12, 1.0, 256))
)

# Resolusi interpolasi
N_INTERP = 500

# Jumlah tick colorbar
N_TICKS = 7


# ============================================================
# 2. PILIHAN FORM MODEL & FILE NETCDF 12 BULAN
# ============================================================

# PILIHAN FORM MODEL:
# Cukup ubah nilai di bawah ini menjadi "D" atau "E" sebelum running.
# Atau bisa langsung lewat terminal: python mappingannual_v3.py D  (atau E)
FORM = "E"

# Dukungan argumen command line (jika pengguna mengoper argumen D / E saat eksekusi)
if len(sys.argv) > 1 and sys.argv[1].upper() in ["D", "E"]:
    FORM = sys.argv[1].upper()
else:
    FORM = FORM.upper()

if FORM not in ["D", "E"]:
    raise ValueError(f"Nilai FORM tidak valid: '{FORM}'. Pilihan yang didukung hanya 'D' atau 'E'.")

# Definisi urutan kronologis 12 bulan beserta tahun dan angka bulan
# (Desember 2021 s.d. November 2022)
MONTH_SPECS = [
    ("December",  2021, 12),
    ("January",   2022, 1),
    ("February",  2022, 2),
    ("March",     2022, 3),
    ("April",     2022, 4),
    ("May",       2022, 5),
    ("June",      2022, 6),
    ("July",      2022, 7),
    ("August",    2022, 8),
    ("September", 2022, 9),
    ("October",   2022, 10),
    ("November",  2022, 11)
]

base_ghi_dir = "/data/mahasiswa/romadhon/perez1/ghi"

month_files = {}
for m_name, yr, m_num in MONTH_SPECS:
    # 1. Jalur standar hirarkis: .../<tahun>/<nama_bulan>/<FORM>/nc/ghi_<tahun>-<bulan>.nc
    primary_path = os.path.join(
        base_ghi_dir,
        str(yr),
        m_name.lower(),
        FORM,
        "nc",
        f"ghi_{yr}-{m_num:02d}.nc"
    )

    # 2. Jalur cadangan (flat fallback jika ada file gabungan di folder ghi/nc)
    alt_path = os.path.join(
        base_ghi_dir,
        "nc",
        f"ghi_{m_name.lower()}_form{FORM}.nc"
    )

    if os.path.isfile(primary_path):
        month_files[m_name] = primary_path
    elif os.path.isfile(alt_path):
        month_files[m_name] = alt_path
    else:
        # Default tetap gunakan format hierarki utama
        month_files[m_name] = primary_path


# ============================================================
# 3. URUTAN BULAN & LABEL
# ============================================================

month_order = [
    "December", "January", "February",
    "March", "April", "May",
    "June", "July", "August",
    "September", "October", "November"
]

month_label = {
    "December": "Dec",
    "January": "Jan",
    "February": "Feb",
    "March": "Mar",
    "April": "Apr",
    "May": "May",
    "June": "Jun",
    "July": "Jul",
    "August": "Aug",
    "September": "Sep",
    "October": "Oct",
    "November": "Nov"
}


# ============================================================
# 4. SHAPEFILE JAWA TIMUR & SEKITARNYA
# ============================================================

shapefile = (
    "/data/mahasiswa/romadhon/perez1/gdam/"
    "gadm41_IDN_2.shp"
)

print("Membaca shapefile...")

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
# 5. BATAS KOORDINAT PETA (DISESUAIKAN PERSIS SEPERTI SEASONS)
# ============================================================

map_minx = 110.8
map_maxx = 116.0

map_miny = -9.0
map_maxy = -5.7


# ============================================================
# 6. OUTPUT
# ============================================================

output_dir = "/data/mahasiswa/romadhon/perez1/peta"
os.makedirs(output_dir, exist_ok=True)

# Nama file output otomatis menyertakan tag Form (D atau E)
output_file = os.path.join(
    output_dir,
    f"peta_ghi_jatim_annual_form{FORM}.pdf"
)


# ============================================================
# 7. FUNGSI MEMPROSES DATA BULANAN (DENGAN INTERPOLASI 3 TAHAP)
# ============================================================

def proses_bulan(file, nama_bulan):
    print("\n" + "=" * 60)
    print(f"MEMPROSES {nama_bulan} [Form {FORM}]")
    print(f"File Input : {file}")
    print("=" * 60)

    if not os.path.exists(file):
        raise FileNotFoundError(
            f"File tidak ditemukan:\n{file}"
        )

    ghi_candidates = ["GHI", "ghi", "GHI_Wm2", "ghi_Wm2"]
    lat_candidates = ["latitude", "lat", "LATITUDE", "LAT"]
    lon_candidates = ["longitude", "lon", "LONGITUDE", "LON"]

    with xr.open_dataset(file) as ds:
        df = ds.to_dataframe().reset_index()

        ghi_column = next((c for c in ghi_candidates if c in df.columns), None)
        if ghi_column is None:
            raise ValueError(
                f"Kolom GHI tidak ditemukan pada {file}.\n"
                f"Kolom tersedia: {df.columns.tolist()}"
            )

        lat_column = next((c for c in lat_candidates if c in df.columns), None)
        if lat_column is None:
            raise ValueError(
                f"Kolom Latitude tidak ditemukan pada {file}.\n"
                f"Kolom tersedia: {df.columns.tolist()}"
            )

        lon_column = next((c for c in lon_candidates if c in df.columns), None)
        if lon_column is None:
            raise ValueError(
                f"Kolom Longitude tidak ditemukan pada {file}.\n"
                f"Kolom tersedia: {df.columns.tolist()}"
            )

        # Ambil subset kolom yang relevan untuk efisiensi RAM
        sub_df = pd.DataFrame({
            "GHI": pd.to_numeric(df[ghi_column], errors="coerce"),
            "latitude": pd.to_numeric(df[lat_column], errors="coerce").round(5),
            "longitude": pd.to_numeric(df[lon_column], errors="coerce").round(5)
        })

    # Cleaning data
    sub_df = sub_df.dropna(subset=["GHI", "latitude", "longitude"])
    sub_df = sub_df[sub_df["GHI"] >= 0]

    # Rata-rata GHI per grid
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
    print(f"Interpolasi cubic : {nama_bulan}")
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
# 8. PROSES 12 BULAN
# ============================================================

print(f"\nMemulai proses 12 bulan untuk Form Model: {FORM}")

hasil = {}
for nama_bulan in month_order:
    hasil[nama_bulan] = proses_bulan(
        month_files[nama_bulan],
        nama_bulan
    )


# ============================================================
# 9. GABUNGKAN SEMUA NILAI (SKALA COLORBAR TERSATU)
# ============================================================

semua_nilai = []
for nama_bulan in month_order:
    nilai = hasil[nama_bulan]["rata"]["GHI"].values
    nilai = nilai[np.isfinite(nilai)]
    semua_nilai.append(nilai)

semua_nilai = np.concatenate(semua_nilai)
vmin = float(np.min(semua_nilai))
vmax = float(np.max(semua_nilai))

print("\n" + "=" * 60)
print(f"SKALA COLORBAR 12 BULAN [FORM {FORM}]")
print("=" * 60)
print(f"Minimum : {vmin:.2f} W/m²")
print(f"Maksimum : {vmax:.2f} W/m²")

norm = Normalize(vmin=vmin, vmax=vmax)
contour_levels = np.linspace(vmin, vmax, 200)


# ============================================================
# 10. INISIALISASI FIGURE & LAYOUT 3 × 4
# ============================================================

fig = plt.figure(figsize=(12, 10))

W = 0.25
H = 0.18

X1 = 0.045
X2 = 0.325
X3 = 0.605

Y1 = 0.070
Y2 = 0.290
Y3 = 0.510
Y4 = 0.730

axes = [
    # Baris 1: Dec | Jan | Feb
    fig.add_axes([X1, Y4, W, H]),
    fig.add_axes([X2, Y4, W, H]),
    fig.add_axes([X3, Y4, W, H]),

    # Baris 2: Mar | Apr | May
    fig.add_axes([X1, Y3, W, H]),
    fig.add_axes([X2, Y3, W, H]),
    fig.add_axes([X3, Y3, W, H]),

    # Baris 3: Jun | Jul | Aug
    fig.add_axes([X1, Y2, W, H]),
    fig.add_axes([X2, Y2, W, H]),
    fig.add_axes([X3, Y2, W, H]),

    # Baris 4: Sep | Oct | Nov
    fig.add_axes([X1, Y1, W, H]),
    fig.add_axes([X2, Y1, W, H]),
    fig.add_axes([X3, Y1, W, H])
]


# ============================================================
# 11. PLOT 12 BULAN
# ============================================================

for ax, nama_bulan in zip(axes, month_order):
    data = hasil[nama_bulan]
    grid_lon_2d = data["grid_lon_2d"]
    grid_lat_2d = data["grid_lat_2d"]
    grid_ghi = data["grid_ghi"]

    # Kunci nilai hasil interpolasi tepat di rentang [vmin, vmax]
    # untuk memastikan tidak ada lubang akibat spline undershoot/overshoot
    grid_ghi_clamped = np.clip(grid_ghi, a_min=vmin, a_max=vmax)

    # Background wilayah pendukung
    background.plot(
        ax=ax,
        facecolor="white",
        edgecolor="black",
        linewidth=0.35,
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
        linewidth=0.40,
        zorder=5
    )

    ax.set_xlim(map_minx, map_maxx)
    ax.set_ylim(map_miny, map_maxy)

    ax.set_xticks(np.arange(111, 117, 1))
    ax.set_yticks(np.arange(-9, -5, 1))

    ax.set_xlabel("Longitude (°)", fontsize=6.5, labelpad=1)
    ax.set_ylabel("Latitude (°)", fontsize=6.5, labelpad=1)
    ax.tick_params(axis="both", labelsize=6, pad=1, length=2)
    ax.set_title(month_label[nama_bulan], fontsize=9, pad=2)
    ax.grid(False)


# ============================================================
# 12. SATU COLORBAR TERPADU (UJUNG DATAR / TIDAK LANCIP)
# ============================================================

sm = ScalarMappable(norm=norm, cmap=CMAP)
sm.set_array([])

cbar_ax = fig.add_axes([0.890, 0.110, 0.018, 0.745])
cbar = fig.colorbar(
    sm,
    cax=cbar_ax,
    ticks=np.linspace(vmin, vmax, N_TICKS),
    extend="neither"
)

cbar.set_label("GHI (W m$^{-2}$)", fontsize=10, labelpad=7)
cbar.ax.tick_params(labelsize=7, length=3, width=0.7)
cbar.minorticks_off()
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))


# ============================================================
# 13. JUDUL UTAMA & SIMPAN PDF
# ============================================================

fig.suptitle(
    f"Global Horizontal Irradiance Jawa Timur (Form {FORM})",
    fontsize=20,
    y=0.970
)

plt.savefig(
    output_file,
    format="pdf",
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 14. SELESAI
# ============================================================

print("\n" + "=" * 60)
print(f"PEMBUATAN PETA 12 BULAN SELESAI [FORM {FORM}]")
print("=" * 60)
print(f"Form Model   : {FORM}")
print(f"Output       : {output_file}")
print("Layout       : 3 × 4 (Dec → Nov)")
print(f"Legenda      : {vmin:.1f} → {vmax:.1f} W/m² (Ujung Datar)")
print(f"Batas peta   : {map_minx}°–{map_maxx}° BT, {map_miny}°–{map_maxy}° LS")
print("Colormap     : Truncated YlOrRd Solid")
print("Interpolasi  : 3-Tahap (cubic -> linear -> nearest) + Clamping")
print("=" * 60)
