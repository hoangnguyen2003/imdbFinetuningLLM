from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_text(prompt, generation_config, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        inputs['input_ids'],
        max_length=generation_config['max_length']
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)