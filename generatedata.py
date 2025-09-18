import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Parameters
start_date = datetime.today() - timedelta(days=180)
days = 180
dates, weights = [], []

current_weight = 200.0  # starting weight

# Pick some gap start indices
gap_starts = set(random.sample(range(0, days-15), 5))

i = 0
while i < days:
    if i in gap_starts:
        i += random.randint(7, 10)  # skip 7–10 days for a gap
        continue
    date = start_date + timedelta(days=i)

    # General downward trend, ~15 lbs over 6 months
    trend = -15.0 * (i / days)
    noise = random.uniform(-1.5, 1.5)
    weight = 200.0 + trend + noise

    dates.append(date.strftime("%Y-%m-%d"))
    weights.append(round(weight, 1))
    i += 1

df = pd.DataFrame({"Date": dates, "Weight": weights})

# Save to current directory
csv_path = "guest_default_dataset.csv"
df.to_csv(csv_path, index=False)

csv_path