import os
import pandas as pd


def load_azt1d_cgm_dataset(base_path):
    all_rows = []

    for root, dirs, files in os.walk(base_path):
        for f in files:
            if f.lower().endswith(".csv"):
                fp = os.path.join(root, f)
                try:
                    df = pd.read_csv(fp)
                    df.columns = df.columns.str.lower()

                    if "cgm" in df.columns:
                        df.rename(columns={"cgm": "glucose"}, inplace=True)

                    if "eventdatetime" in df.columns:
                        df.rename(columns={"eventdatetime": "timestamp"}, inplace=True)

                    if "glucose" in df.columns and "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                        df["glucose"] = pd.to_numeric(df["glucose"], errors="coerce")
                        df["subject_id"] = os.path.basename(root)

                        all_rows.append(df[["subject_id", "timestamp", "glucose"]])

                except:
                    continue

    if not all_rows:
        raise ValueError("No valid CSV files found in dataset.")

    final_df = pd.concat(all_rows, ignore_index=True)
    final_df = final_df.dropna()
    final_df = final_df.sort_values("timestamp")

    return final_df
