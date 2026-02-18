import pandas as pd
import os

input_path = "/mnt/data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_test/train-00000-of-00026_white.parquet"
output_dir = "/mnt/data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_test2"
output_path = os.path.join(output_dir, "train-00000-of-00026_white_1000.parquet")

os.makedirs(output_dir, exist_ok=True)

df = pd.read_parquet(input_path)
print(f"Original rows: {len(df)}")

df_subset = df.head(1000)
df_subset.to_parquet(output_path, index=False)
print(f"Saved {len(df_subset)} rows to {output_path}")