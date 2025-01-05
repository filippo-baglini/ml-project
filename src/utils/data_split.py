import numpy as np

def shuffle_data (input_data: np.ndarray, output_data: np.ndarray):
    """Function to shuffle input and output data, to prevent ordering bias"""

    indices = np.arange(len(input_data))
    np.random.shuffle(indices)

    input_data_shuffled = input_data[indices]
    output_data_shuffled = output_data[indices]

    return input_data_shuffled, output_data_shuffled

def train_val_splitter (input_data: np.ndarray, output_data: np.ndarray, perc_eval_data: float):
    """Function to split input and output data in a training and validation set"""

    input_shuffled, output_shuffled = shuffle_data(input_data, output_data)

    split_index = int(len(input_shuffled) * (1 - perc_eval_data))
    train_input = input_shuffled[:split_index]  
    eval_input = input_shuffled[split_index:]  

    train_output = output_shuffled[:split_index]  
    eval_output = output_shuffled[split_index:]  

    return train_input, eval_input, train_output, eval_output

def train_test_splitter (input_data: np.ndarray, output_data: np.ndarray, perc_test_data: float):
    """Function to split input and output data in a training and test set"""

    input_shuffled, output_shuffled = shuffle_data(input_data, output_data)

    split_index = int(len(input_shuffled) * (1 - perc_test_data))
    train_input = input_shuffled[:split_index]  
    test_input = input_shuffled[split_index:]  

    train_output = output_shuffled[:split_index]  
    test_output = output_shuffled[split_index:]   

    return train_input, test_input, train_output, test_output

def hold_out_splitter (input_data: np.ndarray, output_data: np.ndarray, perc_eval_data: float, perc_test_data: float):
    """Function to split input and output data in a training, validation and test set"""
    
    input_shuffled, output_shuffled = shuffle_data(input_data, output_data)

    # Determine the split indices
    train_split = int(len(input_shuffled) * (1 - (perc_eval_data + perc_test_data)))
    eval_split = int(len(input_shuffled) * (1 - (perc_test_data)))

    # Slice the array
    train_input = input_shuffled[:train_split]  
    eval_input = input_shuffled[train_split:eval_split]
    test_input = input_shuffled[eval_split:]

    train_output = output_shuffled[:train_split]  
    eval_output = output_shuffled[train_split:eval_split]
    test_output = output_shuffled[eval_split:]

    return train_input, eval_input, test_input, train_output, eval_output, test_output

def k_fold_splitter(input_data: np.ndarray, output_data: np.ndarray, folds_number: int):
    """Function to split input and output data in folds, to be used for k-fold cross validation"""

    #input_shuffled, output_shuffled = shuffle_data(input_data, output_data)

    input_folds = np.array_split(input_data, folds_number)
    output_folds = np.array_split(output_data, folds_number)

    return input_folds, output_folds