from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

def setup_training(tokenized_data, tokenizer, training_config, dataset_config):
    train_size = int(dataset_config['train_ratio'] * len(tokenized_data))
    train_data = tokenized_data.shuffle().select(range(train_size))
    eval_data = tokenized_data.shuffle().select(range(train_size, len(tokenized_data)))
    
    model = AutoModelForCausalLM.from_pretrained(training_config.get('model_name', 'distilgpt2'))
    
    training_args = TrainingArguments(
        output_dir=training_config['output_dir'],
        eval_strategy=training_config['evaluation_strategy'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
        num_train_epochs=training_config['num_train_epochs'],
        logging_dir=training_config['logging_dir'],
        logging_steps=training_config['logging_steps'],
        save_total_limit=training_config['save_total_limit']
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data
    )
    
    return trainer, model, tokenizer

def train_and_save(trainer, model, tokenizer, model_path):
    trainer.train()
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)