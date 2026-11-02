import glob
from satpy import Scene

# ============================================================
# PATH DATA HIMAWARI B03
# ============================================================

path = (
    "/data/mahasiswa/romadhon/perez/"
    "himawari/himawari_L1B/"
    "HIMAWARI-8/AHI-L1b-FLDK/"
    "2021/12/01/0000"
)

# ============================================================
# CARI FILE B03
# ============================================================

files = sorted(
    glob.glob(
        path + "/*B03*.DAT.bz2"
    )
)

print("=" * 60)
print("JUMLAH FILE:", len(files))
print("=" * 60)

print(files[:5])

# ============================================================
# BUKA SCENE
# ============================================================

scn = Scene(
    reader="ahi_hsd",
    filenames=files
)

# ============================================================
# LOAD B03 SEBAGAI RADIANCE
# ============================================================

scn.load(
    ["B03"],
    calibration="radiance"
)

data = scn["B03"]

# ============================================================
# METADATA B03
# ============================================================

print("\n" + "=" * 60)
print("METADATA B03")
print("=" * 60)

for key, value in data.attrs.items():
    print(f"{key}: {value}")

# ============================================================
# DATA
# ============================================================

print("\n" + "=" * 60)
print("DATA")
print("=" * 60)

print("dtype :", data.dtype)
print("shape :", data.shape)

print("min   :", data.min().values)
print("mean  :", data.mean().values)
print("max   :", data.max().values)

# ============================================================
# CONTOH NILAI
# ============================================================

print("\n" + "=" * 60)
print("CONTOH DATA")
print("=" * 60)

print(data.values.flatten()[:20])