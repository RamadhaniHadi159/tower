import pandas as pd

file_input = (
    "/data/mahasiswa/romadhon/perez1/"
    "ghi/combined/GHI_djf_season.txt"
)

df = pd.read_csv(file_input, sep="\t")

df["valid_time"] = pd.to_datetime(df["valid_time"])

print("================================")
print("TIMESTAMP CHECK")
print("================================")

print("Jumlah data :", len(df))

print("\nTimestamp pertama:")
print(df["valid_time"].min())

print("\nTimestamp terakhir:")
print(df["valid_time"].max())

print("\nJumlah timestamp unik:")
print(df["valid_time"].nunique())

print("\nJumlah duplikat timestamp:")
print(df["valid_time"].duplicated().sum())

print("\nContoh timestamp:")
print(df["valid_time"].head(20).to_string(index=False))

print("\n================================")