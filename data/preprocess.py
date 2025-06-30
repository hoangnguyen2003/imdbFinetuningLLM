from transformers import AutoTokenizer

def preprocess_text(batch):
    batch['text'] = [text.replace('\n', ' ') for text in batch['text']]
    return batch

def tokenize_data(dataset, model_config, tokenizer=None):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        tokenized = tokenizer(examples['text'], 
                            padding="max_length", 
                            truncation=True, 
                            max_length=model_config['max_length'])
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized
    
    return dataset.map(tokenize_function, batched=True), tokenizer