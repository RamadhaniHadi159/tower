import pandas as pd

file_input = (
    "/data/mahasiswa/romadhon/perez1/"
    "ghi/combined/GHI_djf_season.txt"
)

df = pd.read_csv(
    file_input,
    sep="\t"
)

print("================================")
print("CHECK VALID_TIME")
print("================================")

print("\nTipe data:")
print(df["valid_time"].dtype)

print("\n5 data RAW:")
print(df["valid_time"].head(10).to_string(index=False))

print("\n5 data setelah datetime:")
dt = pd.to_datetime(df["valid_time"])
print(dt.head(10).to_string(index=False))

print("\n================================")
print("FORMAT WAKTU")
print("================================")

print("\nJumlah timestamp unik:")
print(dt.nunique())

print("\nContoh timestamp unik pertama:")
print(
    dt.drop_duplicates()
      .sort_values()
      .head(30)
      .to_string(index=False)
)

print("\n================================")
print("JUMLAH DATA PER JAM")
print("================================")

hour_count = (
    dt.dt.hour
      .value_counts()
      .sort_index()
)

print(hour_count)

print("\n================================")
print("JUMLAH DATA PER TANGGAL")
print("================================")

date_count = (
    dt.dt.date
      .value_counts()
      .sort_index()
)

print(date_count.head(20))

print("\n================================")