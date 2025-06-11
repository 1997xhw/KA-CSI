import numpy as np

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def unwrap_phase(phase):
    return np.unwrap(phase)

def load_numpy_data(root):
    x_amp = np.load(root + 'our_data_amp_1000_270_200.npy', allow_pickle=True)
    x_phase = np.load(root + 'our_data_phase_1000_270_200.npy', allow_pickle=True)
    label = np.load(root + 'our_data_label_1000_270_200.npy', allow_pickle=True)

    x_phase = unwrap_phase(x_phase)
    return x_amp, x_phase, label
