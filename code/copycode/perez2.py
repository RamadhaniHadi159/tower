import os
import pandas as pd
import numpy as np
import xarray as xr


# ============================================================
# LOAD DATA AWAL
# ============================================================

input_file = (
    "/data/mahasiswa/romadhon/"
    "perez1/perez/nc/perez_february.nc"
)


# ============================================================
# BACA NETCDF
# ============================================================

ds_input = xr.open_dataset(input_file)

df = ds_input.to_dataframe().reset_index()

ds_input.close()

print("Jumlah data input :", len(df))
print("Kolom input:")
print(df.columns.tolist())


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

output_nc_dir = (
    "/data/mahasiswa/romadhon/"
    "perez1/ghi/nc/"
)

output_txt_dir = (
    "/data/mahasiswa/romadhon/"
    "perez1/ghi/txt/"
)

os.makedirs(
    output_nc_dir,
    exist_ok=True
)

os.makedirs(
    output_txt_dir,
    exist_ok=True
)

print("\n===== CEK SZA SEBELUM CSAT =====")
print("SZA minimum :", df["sza_deg"].min())
print("SZA maksimum:", df["sza_deg"].max())

print("SZA >= 90° :", (df["sza_deg"] >= 90).sum())
print("SZA >= 80° :", (df["sza_deg"] >= 80).sum())

print("\nContoh SZA >= 90°:")
print(
    df.loc[
        df["sza_deg"] >= 90,
        ["valid_time", "latitude", "longitude", "sza_deg"]
    ].head(20)
)

print("\n===== CEK INPUT CSAT =====")

for col in ["reflectance", "am", "epsilon", "sza_deg"]:

    print(f"\n{col}")
    print("NaN      :", df[col].isna().sum())
    print("+/- inf  :", np.isinf(df[col]).sum())
    print("finite   :", np.isfinite(df[col]).sum())
    print("minimum  :", df[col].min())
    print("maximum  :", df[col].max())


print("\n===== CEK BARIS INVALID =====")

mask_invalid = (
    ~np.isfinite(df["reflectance"]) |
    ~np.isfinite(df["am"]) |
    ~np.isfinite(df["epsilon"]) |
    ~np.isfinite(df["sza_deg"])
)

print("Jumlah baris invalid :", mask_invalid.sum())

print(
    df.loc[
        mask_invalid,
        [
            "valid_time",
            "latitude",
            "longitude",
            "reflectance",
            "am",
            "epsilon",
            "sza_deg"
        ]
    ].head(20)
)

# ============================================================
# 1. HITUNG CSAT
# ============================================================

C_raw = df["reflectance"]
AM = df["am"]
epsilon = df["epsilon"]
theta_z = df["sza_deg"]

df["Csat"] = (
    (C_raw * AM * epsilon) / 2.283
) * ((90 - theta_z) ** (-0.26)) * np.exp(
    0.004 * (90 - theta_z)
)

df["Csat"] = df["Csat"].round(3)

print("\n===== CEK CSAT INVALID =====")

mask_csat_invalid = ~np.isfinite(df["Csat"])

print("Jumlah Csat invalid :", mask_csat_invalid.sum())

print(
    df.loc[
        mask_csat_invalid,
        [
            "valid_time",
            "latitude",
            "longitude",
            "reflectance",
            "am",
            "epsilon",
            "sza_deg",
            "Csat"
        ]
    ].head(20)
)

# ============================================================
# 2. HITUNG CLOUD INDEX (CI)
# ============================================================

Csat_clean = df["Csat"].dropna()

sorted_csat = np.sort(Csat_clean)

print("\n===== STATISTIK CSAT =====")

print(
    "Cmin lama :",
    np.mean(sorted_csat[1:5])
)

print(
    "Cmax lama :",
    np.mean(sorted_csat[-10:])
)

print(
    "P1  :",
    np.percentile(Csat_clean, 1)
)

print(
    "P5  :",
    np.percentile(Csat_clean, 5)
)

print(
    "P50 :",
    np.percentile(Csat_clean, 50)
)

print(
    "P95 :",
    np.percentile(Csat_clean, 95)
)

print(
    "P99 :",
    np.percentile(Csat_clean, 99)
)

print(
    "Csat minimum :",
    np.min(Csat_clean)
)

print(
    "Csat maksimum:",
    np.max(Csat_clean)
)


cmin = np.percentile(
    Csat_clean,
    1
)

cmax = np.percentile(
    Csat_clean,
    99
)

df["CI"] = (
    (df["Csat"] - cmin)
    / (cmax - cmin)
)

df["CI"] = df["CI"].clip(
    lower=0,
    upper=1
)

print("\n===== STATISTIK CLOUD INDEX =====")

print(
    df["CI"].describe()
)

print(
    "\nJumlah CI=0 :",
    (df["CI"] == 0).sum()
)

print(
    "Jumlah CI=1 :",
    (df["CI"] == 1).sum()
)

df["CI"] = df["CI"].round(3)


# ============================================================
# 3. HITUNG LINKE TURBIDITY (TL)
# ============================================================

W = df["tcwv_cm"] / 10

beta = df[
    "angstrom_turbidity"
]

df["TL"] = (
    (
        1.8494
        + 0.2425 * W
        - 0.0203 * W**2
    )
    +
    (
        15.427
        + 0.3153 * W
        - 0.0254 * W**2
    )
    * beta
)

df["TL"] = df["TL"].round(3)


# ============================================================
# 4. HITUNG GHIC
# ============================================================

z = df["elevation_m"]
theta_z = df["sza_deg"]
TL = df["TL"]
AM = df["am"]
I_0 = df["I0_Wm2"]

cos_theta_z = np.cos(
    np.radians(theta_z)
)

cg1 = (
    0.0000509 * z
    + 0.868
)

cg2 = (
    0.0000392 * z
    + 0.0387
)

fh1 = np.exp(
    -z / 8000
)

fh2 = np.exp(
    -z / 1250
)

df["GHIC"] = (
    cg1
    * I_0
    * cos_theta_z
    * np.exp(
        -cg2
        * AM
        * (
            fh1
            + fh2 * (TL - 1)
        )
    )
    * np.exp(
        0.01 * AM**1.8
    )
)

df["GHIC"] = df["GHIC"].round(3)


# ============================================================
# 5. HITUNG kc
# ============================================================

df["kc"] = (
    2.36 * df["CI"]**5
    - 6.2 * df["CI"]**4
    + 6.22 * df["CI"]**3
    - 2.63 * df["CI"]**2
    - 0.58 * df["CI"]
    + 1
)

df["kc"] = df["kc"].round(3)


# ============================================================
# 6. HITUNG GHI FINAL
# ============================================================

df["GHI"] = (
    df["kc"]
    * df["GHIC"]
    * (
        0.0001
        * df["kc"]
        * df["GHIC"]
        + 0.9
    )
)

df["GHI"] = df["GHI"].round(3)


# ============================================================
# 7. OUTPUT TXT
# ============================================================

output_txt = os.path.join(
    output_txt_dir,
    "ghi_february.txt"
)

df.to_csv(
    output_txt,
    sep="\t",
    index=False
)


# ============================================================
# 8. OUTPUT NETCDF
# ============================================================

output_nc = os.path.join(
    output_nc_dir,
    "ghi_february.nc"
)


# ============================================================
# BUAT DATASET NETCDF
# ============================================================

data_vars = {}


# ------------------------------------------------------------
# Variabel hasil model Perez
# ------------------------------------------------------------

output_columns = [
    "Csat",
    "CI",
    "TL",
    "GHIC",
    "kc",
    "GHI"
]

for col in output_columns:

    data_vars[col] = (
        ["obs"],
        df[col].values
    )


# ------------------------------------------------------------
# Variabel input
# ------------------------------------------------------------

input_columns = [
    "reflectance",
    "am",
    "epsilon",
    "sza_deg",
    "tcwv_cm",
    "angstrom_turbidity",
    "elevation_m",
    "I0_Wm2",
    "ssrd_Wm2"
]

for col in input_columns:

    if col in df.columns:

        data_vars[col] = (
            ["obs"],
            df[col].values
        )


# ============================================================
# DATASET
# ============================================================

ds = xr.Dataset(
    data_vars=data_vars,
    coords={
        "obs": np.arange(len(df))
    }
)


# ============================================================
# TAMBAHKAN VALID_TIME
# ============================================================

if "valid_time" in df.columns:

    ds["valid_time"] = (
        "obs",
        pd.to_datetime(
            df["valid_time"],
            errors="coerce"
        ).values
    )

else:

    raise ValueError(
        "Kolom 'valid_time' tidak ditemukan "
        "pada DataFrame input."
    )


# ============================================================
# TAMBAHKAN LATITUDE
# ============================================================

if "latitude" in df.columns:

    ds["latitude"] = (
        "obs",
        df["latitude"].values
    )

    ds["latitude"].attrs["units"] = (
        "degrees_north"
    )


# ============================================================
# TAMBAHKAN LONGITUDE
# ============================================================

if "longitude" in df.columns:

    ds["longitude"] = (
        "obs",
        df["longitude"].values
    )

    ds["longitude"].attrs["units"] = (
        "degrees_east"
    )


# ============================================================
# METADATA
# ============================================================

ds.attrs["title"] = (
    "Global Horizontal Irradiance "
    "Estimated Using Perez Semi-Empirical Model"
)

ds.attrs["model"] = (
    "Perez Semi-Empirical Model"
)

ds.attrs["region"] = (
    "Jawa Timur"
)

ds.attrs["source"] = (
    "Himawari-derived dataset"
)


# ============================================================
# ATTRIBUTE HASIL PEREZ
# ============================================================

ds["GHI"].attrs["long_name"] = (
    "Global Horizontal Irradiance"
)

ds["GHI"].attrs["units"] = (
    "W m-2"
)


ds["GHIC"].attrs["long_name"] = (
    "Clear-sky Global Horizontal Irradiance"
)

ds["GHIC"].attrs["units"] = (
    "W m-2"
)


ds["Csat"].attrs["long_name"] = (
    "Satellite-derived Csat"
)

ds["Csat"].attrs["units"] = "1"


ds["CI"].attrs["long_name"] = (
    "Cloud Index"
)

ds["CI"].attrs["units"] = "1"


ds["TL"].attrs["long_name"] = (
    "Linke Turbidity"
)

ds["TL"].attrs["units"] = "1"


ds["kc"].attrs["long_name"] = (
    "Cloud Modification Factor"
)

ds["kc"].attrs["units"] = "1"


# ============================================================
# ATTRIBUTE INPUT
# ============================================================

if "I0_Wm2" in ds:

    ds["I0_Wm2"].attrs["long_name"] = (
        "Extraterrestrial Irradiance"
    )

    ds["I0_Wm2"].attrs["units"] = (
        "W m-2"
    )


if "sza_deg" in ds:

    ds["sza_deg"].attrs["long_name"] = (
        "Solar Zenith Angle"
    )

    ds["sza_deg"].attrs["units"] = (
        "degree"
    )


if "am" in ds:

    ds["am"].attrs["long_name"] = (
        "Air Mass"
    )

    ds["am"].attrs["units"] = "1"


if "epsilon" in ds:

    ds["epsilon"].attrs["long_name"] = (
        "Earth-Sun Distance Correction"
    )

    ds["epsilon"].attrs["units"] = "1"


if "tcwv_cm" in ds:

    ds["tcwv_cm"].attrs["long_name"] = (
        "Total Column Water Vapour"
    )

    ds["tcwv_cm"].attrs["units"] = "cm"


if "angstrom_turbidity" in ds:

    ds["angstrom_turbidity"].attrs["long_name"] = (
        "Angstrom Turbidity"
    )

    ds["angstrom_turbidity"].attrs["units"] = "1"


if "elevation_m" in ds:

    ds["elevation_m"].attrs["long_name"] = (
        "Elevation"
    )

    ds["elevation_m"].attrs["units"] = "m"


if "reflectance" in ds:

    ds["reflectance"].attrs["long_name"] = (
        "Reflectance"
    )

    ds["reflectance"].attrs["units"] = "1"


if "ssrd_Wm2" in ds:

    ds["ssrd_Wm2"].attrs["long_name"] = (
        "Surface Solar Radiation Downwards"
    )

    ds["ssrd_Wm2"].attrs["units"] = "W m-2"


# ============================================================
# ATTRIBUTE VALID_TIME
# ============================================================

ds["valid_time"].attrs["long_name"] = (
    "Valid Time"
)


# ============================================================
# SIMPAN NETCDF
# ============================================================

ds.to_netcdf(
    output_nc,
    format="NETCDF4"
)


# ============================================================
# RINGKASAN HASIL
# ============================================================

print("\n======================================")
print("        HASIL MODEL PEREZ")
print("======================================")

print(
    "Cmin :",
    round(cmin, 5)
)

print(
    "Cmax :",
    round(cmax, 5)
)

print("\nOutput TXT :")
print(output_txt)

print("\nOutput NC :")
print(output_nc)

print("\n======================================")

print(
    df[
        [
            "Csat",
            "CI",
            "TL",
            "GHIC",
            "kc",
            "GHI"
        ]
    ].head(20)
)

print("======================================")