import numpy as np
import pandas as pd

np.random.seed(42)

SAMPLES = 5000
data = []

# ==================================
# HEALTHY SIGNALS (40%)
# ==================================
for _ in range(int(SAMPLES * 0.40)):
    data.append([
        np.random.uniform(-14, -9),          # Power
        np.random.uniform(0.005, 0.02),      # Noise
        10 ** np.random.uniform(-11, -9),    # BER
        np.random.uniform(14, 18),            # Q-factor
        np.random.choice([1540, 1550, 1552, 1555]),
        np.random.uniform(5, 40),             # Distance
        0                                     # HEALTHY
    ])

# ==================================
# DEGRADED SIGNALS (35%)
# ==================================
for _ in range(int(SAMPLES * 0.35)):
    data.append([
        np.random.uniform(-18, -13),          # Power
        np.random.uniform(0.02, 0.05),        # Noise
        10 ** np.random.uniform(-8, -6),      # BER
        np.random.uniform(8, 13),              # Q-factor
        np.random.choice([1540, 1550, 1552, 1555]),
        np.random.uniform(40, 80),             # Distance
        1                                     # DEGRADED
    ])

# ==================================
# CRITICAL SIGNALS (25%)
# ==================================
for _ in range(int(SAMPLES * 0.25)):
    data.append([
        np.random.uniform(-25, -18),           # Power
        np.random.uniform(0.05, 0.12),         # Noise
        10 ** np.random.uniform(-6, -3),       # BER
        np.random.uniform(3, 8),                # Q-factor
        np.random.choice([1540, 1550, 1552, 1555]),
        np.random.uniform(80, 120),             # Distance
        2                                     # CRITICAL
    ])

columns = [
    "power",
    "noise",
    "ber",
    "q_factor",
    "wavelength",
    "distance",
    "label"
]

df = pd.DataFrame(data, columns=columns)

# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

# Save
df.to_csv("data/fiber_data.csv", index=False)

print("✅ 5,000-sample multi-class fiber dataset generated")
print(df["label"].value_counts())
