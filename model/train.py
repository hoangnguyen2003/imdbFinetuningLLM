from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

def setup_training(tokenized_data, train_ratio, model_name, training_config):
    train_size = int(train_ratio * len(tokenized_data))
    train_data = tokenized_data.shuffle().select(range(train_size))
    eval_data = tokenized_data.shuffle().select(range(train_size, len(tokenized_data)))
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    training_args = TrainingArguments(
        output_dir=training_config['output_dir'],
        eval_strategy='epoch',
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
        num_train_epochs=training_config['num_train_epochs'],
        logging_steps=10,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data
    )
    
    return model, trainer

def train_and_save(model_path, tokenizer, model, trainer):
    trainer.train()
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)