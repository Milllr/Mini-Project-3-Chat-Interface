import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

# Initialize the tokenizer and model with the token for Model 1 (DialoGPT)
tokenizer_1 = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
model_1 = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl").to('cpu')

# Initialize the tokenizer and model for Model 2 (FLAN-T5)
tokenizer_2 = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
model_2 = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl").to('cpu')

'''Tweaked your generate response function so it takes in the context of someones background when replying:'''
def generate_response(model, tokenizer, prompt, context=None, max_tokens=100, temperature=0.99):
    """
    (changed temp to 0.99 to make it more interesting | im sure you know how temperature influences the output of an LLM)

    Generates a response from the model based on the given prompt and optional context.

    Args:
        model: The pre-trained model used for generating responses.
        tokenizer: The tokenizer corresponding to the model.
        prompt (str): The input prompt for the model.
        context (str, optional): Context string to influence the model's response.
        max_tokens (int): Maximum number of tokens for the model's output.
        temperature (float): Sampling temperature for the model's generation.

    Returns:
        str: The model's generated response.
    """
    # Prepend context to the prompt if provided
    if context:
        prompt = f"{context}\n\n{prompt}"

    # Tokenize and generate response
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    outputs = model.generate(
        inputs['input_ids'], 
        max_new_tokens=max_tokens, 
        temperature=temperature, 
        do_sample=True,
        attention_mask=inputs['attention_mask'],
        pad_token_id=tokenizer.eos_token_id  # Set `pad_token_id` explicitly
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


'''NEW FUNCTION'''
def generate_context_with_model(model, tokenizer, political_leaning, age, wealth_level, max_tokens=100, temperature=0.7):
    """
    Generates a context paragraph by calling a model based on political leaning, age, and wealth level.

    Args:
        model: The pre-trained model used to generate the context paragraph.
        tokenizer: The tokenizer corresponding to the model.
        political_leaning (float): A value from -1 (most liberal) to 1 (most conservative).
        age (int): Age of the individual.
        wealth_level (float): A value from 0 (poorest) to 1 (wealthiest).
        max_tokens (int): Maximum number of tokens for the model's output.
        temperature (float): Sampling temperature for the model's generation.

    Returns:
        str: A generated context paragraph.
    """
    # Validate inputs
    if not (-1 <= political_leaning <= 1):
        raise ValueError("Political leaning must be between -1 and 1.")
    if not (0 <= wealth_level <= 1):
        raise ValueError("Wealth level must be between 0 and 1.")
    if age < 0:
        raise ValueError("Age must be a non-negative integer.")

    # Prepare the input prompt for the model
    prompt = (
        f"Generate a context paragraph for an individual based on the following parameters: "
        f"political leaning: {political_leaning} (-1 is most liberal, 1 is most conservative), "
        f"age: {age}, and wealth level: {wealth_level} (0 is poorest, 1 is wealthiest). "
        f"Ensure the paragraph is descriptive and considers socioeconomic and ideological factors."
    )

    # Call the model to generate the context
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    outputs = model.generate(
        inputs['input_ids'],
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        attention_mask=inputs['attention_mask'],
        pad_token_id=tokenizer.eos_token_id
    )
    context = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return context


# Start with an initial prompt
conversation_prompt = "how to prevent from school bullying?"

# Loop indefinitely for a conversation between two models
while True:
    print("Model 1:", conversation_prompt)
    response_1 = generate_response(model_1, tokenizer_1, conversation_prompt)
    print("Model 2:", response_1)
    
    response_2 = generate_response(model_2, tokenizer_2, response_1)
    conversation_prompt = response_2
