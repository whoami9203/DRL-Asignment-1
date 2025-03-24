import pickle
import numpy as np

# Load using NumPy 2.0.2
with open("random10_table.pkl", "rb") as f:
    q_table = pickle.load(f)

# Convert Q-table values to ensure compatibility
q_table_str_keys = {str(key): value for key, value in q_table.items()}

# Save the Q-table to .npy format
np.savez('q_table.npz', **q_table_str_keys)

data = np.load('q_table.npz', allow_pickle=True)
q_table_loaded = {key: data[key] for key in data}
print(f"Loaded Q-table from .npz: {q_table_loaded}")