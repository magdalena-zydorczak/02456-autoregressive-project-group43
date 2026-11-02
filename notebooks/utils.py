from protgen.data import ProteinDataset
from torch.utils.data import DataLoader

def load_data(data_dir_path, type="train", task="classification"):
    if type == "train":
        data = ProteinDataset(path_to_data=f"{data_dir_path}/train.csv")
    elif type == "val":
        data = ProteinDataset(path_to_data=f"{data_dir_path}/val.csv")
    elif type == "test":
        data = ProteinDataset(path_to_data=f"{data_dir_path}/test.csv")
    else:
        raise ValueError("Invalid data type. Choose from 'train', 'val', or 'test'.")

    # Get tensors
    X = data.tokens
    if task == "classification":
        y_labels = data.labels          # classification task
    elif task == "generative":
        y_labels = data.tokens_shifted     # generative task
    else:
        raise ValueError("Invalid task type. Choose from 'classification' or 'generative'.")

    print(f"# {type} data, task: {task}\n Tokens shape: {X.shape}\n Labels (shape): {y_labels.shape}")
    
    loader = DataLoader(data, batch_size=32, shuffle=True)
    
    return data, loader, X, y_labels

def translate_generated_sequence(generated_sequence, tokenizer):
    # Decode the token indices back to amino acid sequence
    amino_acid_sequence = tokenizer.decode(generated_sequence)
    
    return amino_acid_sequence