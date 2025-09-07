from pathlib import Path
import numpy as np, pandas as pd, datetime as dt

DATA = Path(__file__).resolve().parent / "data"
DATA.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)
n = 2000
df = pd.DataFrame({
    "user_id": np.arange(n),
    "mean_f": rng.normal(0,1,n),
    "std_f": np.abs(rng.normal(1,0.2,n)),
    "label": rng.integers(0,2,n),
    "event_timestamp": pd.to_datetime("now") - pd.to_timedelta(rng.integers(0, 3600*24, n), unit="s")
})
df.to_parquet(DATA / "demo.parquet")
print("Wrote", DATA / "demo.parquet")
