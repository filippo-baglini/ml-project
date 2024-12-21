import numpy as np

def shuffle_data (input_data: np.ndarray, output_data: np.ndarray):

    indices = np.arange(len(input_data))
    np.random.shuffle(indices)

    input_data = input_data[indices]
    output_data = output_data[indices]

def train_val_splitter (input_data: np.ndarray, output_data: np.ndarray, perc_eval_data: float):

    indices = np.arange(len(input_data))
    np.random.shuffle(indices)

    input_data = input_data[indices]
    output_data = output_data[indices]

    split_index = int(len(input_data) * (1 - perc_eval_data))
    train_input = input_data[:split_index]  
    val_input = input_data[split_index:]  

    train_output = output_data[:split_index]  
    val_output = output_data[split_index:]  

    return train_input, val_input, train_output, val_output

def train_test_splitter (input_data: np.ndarray, output_data: np.ndarray, perc_test_data: float):

    indices = np.arange(len(input_data))
    np.random.shuffle(indices)

    input_data = input_data[indices]
    output_data = output_data[indices]

    split_index = int(len(input_data) * (1 - perc_test_data))
    train_input = input_data[:split_index]  
    test_input = input_data[split_index:]  

    train_output = output_data[:split_index]  
    test_output = output_data[split_index:]  

    return train_input, test_input, train_output, test_output

def hold_out_splitter (input_data: np.ndarray, output_data: np.ndarray, perc_eval_data: float, perc_test_data: float):
    
    indices = np.arange(len(input_data))
    np.random.shuffle(indices)

    input_data = input_data[indices]
    output_data = output_data[indices]

    # Determine the split indices
    train_split = int(len(input_data) * (1 - (perc_eval_data + perc_test_data)))
    eval_split = int(len(input_data) * (1 - (perc_test_data)))

    # Slice the array
    train_input = input_data[:train_split]  
    eval_input = input_data[train_split:eval_split]
    test_input = input_data[eval_split:]

    train_output = output_data[:train_split]  
    eval_output = output_data[train_split:eval_split]
    test_output = output_data[eval_split:]

    return train_input, eval_input, test_input, train_output, eval_output, test_output

def k_fold_splitter(input_data: np.ndarray, output_data: np.ndarray, folds_number: int):

    input_folds = np.array_split(input_data, folds_number)
    output_folds = np.array_split(output_data, folds_number)

    return input_folds, output_folds