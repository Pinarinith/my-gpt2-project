import torch
from torch.utils.data import Dataset, DataLoader, random_split

class ShakespeareDataset(Dataset):
    def __init__(self, text, char_to_idx, idx_to_char, block_size):
        self.text = text
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.block_size = block_size
        self.vocab_size = len(char_to_idx)

    def __len__(self):
        # We can create (len(self.text) - self.block_size) training examples
        return len(self.text) - self.block_size

    def __getitem__(self, idx):
        # Get a chunk of text
        chunk = self.text[idx:idx + self.block_size + 1]
        # Convert characters to indices
        input_indices = [self.char_to_idx[char] for char in chunk[:-1]]
        target_indices = [self.char_to_idx[char] for char in chunk[1:]]
        return torch.tensor(input_indices, dtype=torch.long), torch.tensor(target_indices, dtype=torch.long)

def prepare_dataset(filepath, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, block_size=128):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # Create character-level vocabulary
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    full_dataset = ShakespeareDataset(text, char_to_idx, idx_to_char, block_size)

    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset, char_to_idx, idx_to_char

