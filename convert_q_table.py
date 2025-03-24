import pickle
import numpy as np

# Load using NumPy 2.0.2
with open("random10_table.pkl", "rb") as f:
    q_table = pickle.load(f)

# Convert Q-table values to ensure compatibility
if isinstance(q_table, dict):
    for key in q_table:
        q_table[key] = np.array(q_table[key], dtype=np.float64)  # Ensure it's compatible

with open("q_table_v1_24.pkl", "wb") as f:
    pickle.dump(q_table, f, protocol=4)  # Protocol 4 is safe for NumPy 1.24.4