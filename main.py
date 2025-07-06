import yaml
import argparse
from data.load_data import load_data
from data.preprocess import preprocess_text, tokenize_data
from model.train import setup_training, train_and_save
from model.generate import generate_text

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(args):
    config = load_config()
    
    if args.mode == 'train':
        dataset = load_data(config['dataset'])
        dataset = dataset.map(preprocess_text, batched=True)
        tokenized_data, tokenizer = tokenize_data(dataset, config['model'])

        model, trainer = setup_training(tokenized_data,
                                        config['dataset']['train_ratio'],
                                        config['model']['name'], config['training'])
        train_and_save(config['training']['model_path'], tokenizer, model, trainer)
    else:
        generated_text = generate_text(args.prompt,
                                       config['generation']['max_length'],
                                       config['training']['model_path'])
        print(generated_text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fine-tuning LLM'
    )
    parser.add_argument(
        '--mode',
        required=True,
        choices=['train', 'generate'],
        help='Mode to run: "train" or "generate"'
    )
    parser.add_argument(
        '--prompt',
        type=str, 
        help='Text prompt for generation mode'
    )

    main(parser.parse_args())