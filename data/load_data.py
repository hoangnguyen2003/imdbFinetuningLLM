from datasets import load_dataset

def load_data(dataset_config):
    return load_dataset(dataset_config['name'], split=dataset_config['split'])