import pandas as pd
import json

df = pd.read_csv('battery_health_dataset.csv')

info = {
    "shape": list(df.shape),
    "columns": df.columns.tolist(),
    "dtypes": {c: str(df[c].dtype) for c in df.columns},
    "nan_total": int(df.isnull().sum().sum()),
    "SoH_min": float(df['SoH'].min()),
    "SoH_max": float(df['SoH'].max()),
    "SoH_mean": float(df['SoH'].mean()),
    "batteries": sorted(df['battery_id'].unique().tolist()),
    "cycles_per_battery": df.groupby('battery_id')['cycle_number'].nunique().to_dict(),
    "bins_per_cycle_sample": df.groupby(['battery_id','cycle_number']).size().describe().to_dict(),
    "head": df.head(5).to_dict(orient='records')
}

with open('inspect_result.json', 'w') as f:
    json.dump(info, f, indent=2, default=str)

print("Done - see inspect_result.json")
