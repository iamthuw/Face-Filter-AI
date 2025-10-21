from sklearn.model_selection import train_test_split

def split_dataset(all_data, test_size=0.2, seed=42):
    """Chia dataset thành train và test."""
    return train_test_split(all_data, test_size=test_size, random_state=seed)
