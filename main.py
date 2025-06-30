import yaml
from data.load_data import load_imdb_dataset
from data.preprocess import preprocess_text, tokenize_data
from model.train import setup_training, train_and_save
from model.generate import generate_text

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    dataset = load_imdb_dataset(config['dataset'])
    
    dataset = dataset.map(preprocess_text, batched=True)
    tokenized_data, tokenizer = tokenize_data(dataset, config['model'])

    trainer, model, tokenizer = setup_training(tokenized_data, tokenizer, config['training'], config['dataset'])
    train_and_save(trainer, model, tokenizer, config['training']['model_path'])
    
    prompt = "The script"
    generated_text = generate_text(prompt, config['generation'], config['training']['model_path'])
    print(generated_text)

if __name__ == "__main__":
    main()