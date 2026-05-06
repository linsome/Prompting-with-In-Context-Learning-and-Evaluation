import json
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import matplotlib.pyplot as plt

print("PyTorch :", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 1. device 
model_name = './Qwen3-0.6B-Base' 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, attn_implementation="eager").to(device)

# 2. read data
with open("C:\\Users\\linso\\Downloads\\code-2\\code\\DATA\\SST2\\train.json", 'r') as f:
    train_data = json.load(f)
with open("C:\\Users\\linso\\Downloads\\code-2\\code\\DATA\\SST2\\val.json", 'r') as f:
    val_data = json.load(f)

# 3. Dynamic Prompt Construction (K-shot Prompting)
def create_k_shot_prompt(target_input, examples, k):
    label_map = {0: "negative", 1: "positive"}
    """
    Example format:
    sentence: [Text]
    label: [Label]
    ...
    sentence: [Target_Text]
    label: 
    """
    prompt = ""
    # k samples from training set
    sampled_examples = random.sample(examples, k)
    for ex in sampled_examples:
        word_label = label_map[ex['label']]
        prompt += f"Input: {ex['sentence']}\Sentiment: {word_label}\n\n"
    
    # val sample
    prompt += f"Input: {target_input}\nSentiment:"
    return prompt

# 4. Parsing Challenge 
def parse_label(generated_text):
    """
    str to num
    """
    first_line = generated_text.split('\n')[0].strip().lower()
    
    if "positive" in first_line:
        return 1   # Positive
    elif "negative" in first_line:
        return 0   # Negative
    return -1      # -1  (unknown)

# 5. Evaluation Loop
def evaluate(val_set, train_set, k=3):
    correct = 0
    
    # only get a subset of validation set for quick evaluation
    # val_subset = val_set[:max_eval]
    # use the whole set
    val_subset = val_set
    total = len(val_subset)
    
    print(f"Starting evaluation with {total} data points...")
    
    for i, item in enumerate(val_subset):
        prompt = create_k_shot_prompt(item['sentence'], train_set, k)
        
        # .to(device) for gpu
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=2, 
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # prediction is 0 or 1, -1 means unknown
        prediction = parse_label(generated_text)
        
        # compare with target label
        if prediction == item['label']:
            correct += 1
            
        # print progress
        # print(f"[{i+1}/{total}] Target: {item['label']} | Pred: {prediction} | Gen: '{generated_text.strip()}'")

    print("-" * 30)
    print(f"Accuracy: {correct / total:.2%}")
    accuracy = correct / total
    return accuracy


k_values = [0, 1, 3, 5, 8, 16]
accuracies = []

print("🚀 Starting batch evaluation...")

for k in k_values:
    print(f"\n" + "="*40)
    print(f"Testing K = {k} ...")
    
    acc = evaluate(val_data, train_data, k=k)
    accuracies.append(acc * 100)
print("\n All tests completed!")

plt.figure(figsize=(8, 6))

# --- Plot the accuracy line chart ---
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8)

# Set title and labels
plt.title('Accuracy vs. K-shots', fontsize=14)
plt.xlabel('Number of Shots (K)', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)

# Add grid lines for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Annotate the specific percentage numbers on the chart
for i, txt in enumerate(accuracies):
    plt.annotate(f"{txt:.1f}%", 
                 (k_values[i], accuracies[i]), 
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()