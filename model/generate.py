from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_text(prompt, max_length_generation, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    input = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        input['input_ids'],
        max_length=max_length_generation
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)