import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Choose the model name
MODEL_NAME = "google/flan-t5-xl"  # Yimin, you should change this to "google/flan-t5-xxl" if you can run it on the gpu rack

# Initialize the tokenizer and model for Model 1
tokenizer_1 = AutoTokenizer.from_pretrained(MODEL_NAME)
model_1 = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to('cuda' if torch.cuda.is_available() else 'cpu') #running off my laptop without a cuda gpu

def map_political_leaning(leaning):
    """
    Maps a numerical political leaning to a detailed descriptive text.
    """
    if leaning <= -0.9:
        return "a far-left progressive activist who strongly advocates for radical changes to achieve social equality"
    elif -0.9 < leaning <= -0.7:
        return "a staunch liberal who passionately supports progressive policies and social justice"
    elif -0.7 < leaning <= -0.5:
        return "a liberal-minded individual who favors progressive reforms and equality"
    elif -0.5 < leaning <= -0.3:
        return "a moderate liberal who leans towards progressive ideas but values some traditional views"
    elif -0.3 < leaning < 0:
        return "a slightly liberal person with progressive tendencies"
    elif leaning == 0:
        return "a centrist who balances liberal and conservative views"
    elif 0 < leaning <= 0.3:
        return "a slightly conservative person with traditional tendencies"
    elif 0.3 < leaning <= 0.5:
        return "a moderate conservative who leans towards traditional values but is open to some progressive ideas"
    elif 0.5 < leaning <= 0.7:
        return "a conservative-minded individual who favors traditional policies and values"
    elif 0.7 < leaning <= 0.9:
        return "a staunch conservative who passionately supports traditional values and policies"
    elif leaning > 0.9:
        return "a far-right traditionalist who strongly advocates for preserving established norms and returning to earlier societal structures"


def map_wealth_level(wealth):
    """
    Maps a numerical wealth level to a detailed descriptive text.
    """
    if wealth <= 0.1:
        return "living in poverty, struggling to meet basic needs"
    elif 0.1 < wealth <= 0.3:
        return "a low-income individual facing financial challenges"
    elif 0.3 < wealth <= 0.5:
        return "a working-class person earning a modest income"
    elif 0.5 < wealth <= 0.7:
        return "a middle-class individual with a comfortable but not extravagant lifestyle"
    elif 0.7 < wealth <= 0.9:
        return "an affluent person enjoying significant financial comfort"
    elif wealth > 0.9:
        return "a wealthy individual with substantial financial resources and luxury"


def generate_context(political_leaning, age, wealth_level):
    """
    Generates a detailed context paragraph based on political leaning, age, and wealth level.
    Emphasizes that the bot must act from the specified viewpoint.
    """
    leaning_desc = map_political_leaning(political_leaning)
    wealth_desc = map_wealth_level(wealth_level)
    context = (f"You are {leaning_desc}, {age} years old, and {wealth_desc}. "
               f"You must engage in the debate by presenting arguments that align with your political beliefs and socio-economic background. "
               f"Your responses should reflect your perspective and experiences as someone from this background.")
    return context


def generate_response(model, tokenizer, prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

def build_prompt(context, debate_topic, conversation_history, speaker, opponent):
    prompt = f"{context}\n\n"
    prompt += f"Debate Topic: {debate_topic}\n\n"
    prompt += f"As a {speaker}, present a comprehensive and persuasive argument supporting your viewpoint on this topic. Address the points made by the {opponent} and provide evidence, examples, and reasoning to strengthen your position.\n\n"
    prompt += "The debate so far:\n"
    for turn in conversation_history:
        prompt += f"{turn['speaker']}: {turn['text']}\n\n"
    prompt += f"{speaker}:"
    return prompt

# Generate contexts for both personas
context_1 = generate_context(
    political_leaning=-1,  # Liberal
    age=40,
    wealth_level=0.5
)

context_2 = generate_context(
    political_leaning=1,  # Conservative
    age=40,
    wealth_level=0.5
)

# Define the debate topic
debate_topic = "Should abortion remain a legal right for all individuals?"

# Initialize conversation history
conversation_history = []

# Number of turns in the conversation
num_turns = 5  # Adjust as needed

# Maximum number of new tokens to generate
max_new_tokens = 300  # Adjust as needed

for turn in range(num_turns):
    print(f"\nTurn {turn + 1}:\n")
    
    # Model 1's turn
    speaker_1 = "Liberal"
    opponent_1 = "Conservative"
    prompt_1 = build_prompt(context_1, debate_topic, conversation_history, speaker_1, opponent_1)
    response_1 = generate_response(model_1, tokenizer_1, prompt_1, max_new_tokens=max_new_tokens)
    print(f"Model 1 ({speaker_1}):")
    print(response_1)
    conversation_history.append({'speaker': speaker_1, 'text': response_1})
    
    # Model 2's turn
    speaker_2 = "Conservative"
    opponent_2 = "Liberal"
    prompt_2 = build_prompt(context_2, debate_topic, conversation_history, speaker_2, opponent_2)
    response_2 = generate_response(model_1, tokenizer_1, prompt_2, max_new_tokens=max_new_tokens)
    print(f"\nModel 2 ({speaker_2}):")
    print(response_2)
    conversation_history.append({'speaker': speaker_2, 'text': response_2})
    
    # Trim conversation history if necessary
    total_tokens = sum([len(tokenizer_1.encode(turn['text'])) for turn in conversation_history])
    while total_tokens > 512 - 200:
        conversation_history.pop(0)
        total_tokens = sum([len(tokenizer_1.encode(turn['text'])) for turn in conversation_history])
