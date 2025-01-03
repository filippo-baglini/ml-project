import numpy as np
import os

def read_monk_data(filename):
    try:
        # Read the file and process each line
        with open(filename, "r") as file:
            lines = file.readlines()
        
        # Use list comprehensions to parse the data
        # Monk Data has for each line: 1 element output target, last element data_index, rest input data
        data = [list(map(int, line.strip().split()[0:-1])) for line in lines]
        
        # Convert to NumPy arrays
        input_data = np.array([values[1:] for values in data])
        output_data = np.array([values[0] for values in data])
        
        return input_data, output_data
    
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


def feature_one_hot_encoding(y, classes):
    '''
    y: array of labels
    classes: array of number of classes
    '''
    one_hot_len = np.sum(classes)
    one_hot = np.zeros((y.shape[0], one_hot_len))
    for i in range(y.shape[0]): #for each sample
        for j in range(len(classes)): #for each label
            #if y has a single dimension, then we have a single output
            if len(classes) == 1:
                one_hot[i, y[i] - 1] = 1
            else:
                prev_classes = int(np.sum(classes[0:j]))
                one_hot[i, prev_classes + y[i,j] - 1] = 1
    return one_hot


def readTrainingCupData(filename:str):
    input = []
    output = []
    with open(filename, "r") as file:
        for line in file:
            if line.startswith('#'): continue
            values = list(map(float, line.split(',')[1:]))
            input.append(values[0:-3])
            output.append(values[-3:])
    return np.array(input), np.array(output)


def readTestCupData(filename:str):
    input = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('#'): continue
            values = list(map(float, line.split(',')[1:]))
            input.append(values)
    return input

def save_predictions(y_pred, folder = "."):
    # Assicurati che `y_pred` abbia 3 colonne
    if y_pred.shape[1] != 3:
        raise ValueError(f"Le predizioni devono avere esattamente 3 colonne (out_x, out_y, out_z). Trovato: {y_pred.shape[1]}")

    # Prepara gli ID e combina i dati
    ids = np.arange(1, y_pred.shape[0] + 1).reshape(-1, 1)  # ID da 1 a n interi
    results = np.hstack((ids, y_pred))

    # Prepara l'header personalizzato
    header = [
        "# Benedetti Gabriele, Filippo Baglini",
        "# Life Is Life",
        "# ML-CUP24 v1",
        "# 22 Jan 2024"
    ]

    # Nome del file di output
    output_file = f"{folder}/life-is-life_ML-CUP24-TS.csv"

    # Scrive il file CSV con l'header personalizzato
    with open(output_file, "w") as f:
        f.write("\n".join(header) + "\n")
        np.savetxt(f, results, fmt="%d,%f,%f,%f", delimiter=",", newline="\n")

    return os.path.abspath(output_file)


def custom_serializer(obj):
    """
    Convert non-serializable objects to JSON-compatible types using __str__ for string representation.
    """
    if hasattr(obj, "__str__"):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable.")