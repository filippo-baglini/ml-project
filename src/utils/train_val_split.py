import numpy as np

def train_val_splitter (input_data: np.ndarray, output_data: np.ndarray, perc_val_data: float):

    indices = np.arange(len(input_data))
    np.random.shuffle(indices)

    input_data = input_data[indices]
    output_data = output_data[indices]

    split_index = int(len(input_data) * (1 - perc_val_data))
    train_input = input_data[:split_index]  
    val_input = input_data[split_index:]  

    train_output = output_data[:split_index]  
    val_output = output_data[split_index:]  

    return train_input, val_input, train_output, val_output

def hold_out_splitter (input_data: np.ndarray, output_data: np.ndarray, perc_val_data: float, perc_test_data: float):
    pass