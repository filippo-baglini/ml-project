import numpy as np

def train_val_splitter(input_data: np.ndarray, output_data: np.ndarray, perc_val_data: float):

    indices = np.arange(len(input_data))
    np.random.shuffle(indices)

    input_data = input_data[indices]
    output_data = output_data[indices]

    split_index = int(len(input_data) * (1 - perc_val_data))
    train_input = input_data[:split_index]  # First 70%
    val_input = input_data[split_index:]  # Last 30%

    train_output = output_data[:split_index]  # First 70%
    val_output = output_data[split_index:]  # Last 30%

    return train_input, val_input, train_output, val_output
