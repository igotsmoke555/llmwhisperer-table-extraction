from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
from tqdm import tqdm
import torch


def load_data(data_path):
    # Load the dataset from the JSON file
    dataset = load_dataset('json', data_files={'train': data_path})

    # Extract relevant data
    for sample in dataset['train']:
        premise = sample['wiki_bio_text']
        hypo_sentences = sample['gpt3_sentences']
        annotations = sample['annotation']

        # Ensure that the number of sentences matches the number of annotations
        if len(hypo_sentences) != len(annotations):
            print("Mismatch in lengths of gpt3_sentences and annotation.")
            continue  # Skip this sample if lengths don't match

        for hypo, annotation in zip(hypo_sentences, annotations):
            if annotation == "major_inaccurate" or annotation == "minor_inaccurate":
                label = 1
            else:
                label = 0
            data.append({
                "premise": premise,
                "hypothesis": hypo,
                "gold_label": label
            })


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device).eval()
    return model, tokenizer


def create_prompt_few(premise, hypothesis):
    return (
        """
        You are a highly accurate natural language inference model. Your task is to analyze the relationship between two sentences: a "premise" and a "hypothesis". Based on their meaning, you must classify the relationship into one of the following categories:       
        1. **Entailment**: The hypothesis logically follows from the premise.
        2. **Contradiction**: The hypothesis contradicts the premise.

        For each input, carefully read the premise and hypothesis, consider their meanings, and return the correct label: "entailment" or "contradiction". The correct label formed as <label>. 
        Be precise and provide the classification as the output.
        """
        "Premise: {premise}. Hypothesis: {hypothesis}. "
        """Task: Determine the relationship between the premise and hypothesis. The possible relationships are:
        - **entailment**: The hypothesis must be true if the premise is true.
        - **contradiction**: The hypothesis cannot be true if the premise is true.

        Please give your result and make sure your output format is:
        - Conclusion: <label>.
        """
    ).format(premise=premise, hypothesis=hypothesis)


def create_prompt_zero(premise, hypothesis):
    return (
        """
        You are a highly accurate natural language inference model. Your task is to analyze the relationship between two sentences: a "premise" and a "hypothesis". Based on their meaning, you must classify the relationship into one of the following categories:       
        For each input, carefully read the premise and hypothesis, consider their meanings, and return the correct label: "entailment" or "contradiction". The correct label formed as <label>. 
        Be precise and provide the classification as the output.
        """
        "Premise: {premise}. Hypothesis: {hypothesis}. "
        """
        Please give your result and make sure your output format is:
        - Conclusion: <label>.
        """
    ).format(premise=premise, hypothesis=hypothesis)


def create_prompt_CoT(premise, hypothesis):
    return (
        """
        You are a highly accurate natural language inference model. Your task is to analyze the relationship between two sentences: a "premise" and a "hypothesis". Based on their meaning, you must classify the relationship into one of the following categories:       
        1. **Entailment**: The hypothesis logically follows from the premise.
        2. **Contradiction**: The hypothesis contradicts the premise.

        For each input, carefully read the premise and hypothesis, consider their meanings, and return the correct label: "entailment" or "contradiction". The correct label formed as <label>. 
        Be precise and provide the classification as the output.
        """
        "Premise: {premise}. Hypothesis: {hypothesis}. "
        """Task: Determine the relationship between the premise and hypothesis. The possible relationships are:
        - **entailment**: The hypothesis must be true if the premise is true.
        - **contradiction**: The hypothesis cannot be true if the premise is true.

        Please provide your answer in the following format:
        - Premise:
        - Hypothesis:
        - Reasoning:
        - Conclusion:

        Examples:
        1. Premise: "She handed him the umbrella because it was raining."
        Hypothesis: "It was raining outside."
        Reasoning: The premise explicitly states that it was raining. Thus, the hypothesis is true if the premise is true.
        Conclusion: entailment.

        2. Premise: "He is allergic to peanuts."
        Hypothesis: "He can eat peanut butter sandwiches."
        Reasoning: If he is allergic to peanuts, he cannot eat peanut butter sandwiches. The two statements cannot be true simultaneously.
        Conclusion: contradiction.

        Please give your result and make sure your output format is:
        - Conclusion: <label>.
        """
    ).format(premise=premise, hypothesis=hypothesis)


def tokenize_prompts(prompt, tokenizer):
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                           add_generation_prompt=True,
                                           tokenize=True,
                                           return_tensors="pt",
                                           return_dict=True
                                           )
    return inputs


def generate_responses(prompt, model, tokenizer, batch_size=1):
    inputs = prompt
    output = model.generate(**inputs, **gen_kwargs)
    generated_id = output[0]
    batch_generated_texts = tokenizer.batch_decode(generated_id, skip_special_tokens=True)
    generated_texts.extend(batch_generated_texts)
    return generated_texts


def map_response_to_label(response):
    # response = response.lower()
    response = response[-30:]
    combined_string = ''.join(response)
    # print(combined_string)
    if 'entailment' in combined_string:
        return 0
    # elif 'contradiction' in combined_string:
    else:
        return 1


device = "cuda"
batch_size = 1
data = []
generated_texts = []
predicted_labels = []
record_responses = []
gen_kwargs = {"max_length": 2000, "do_sample": True, "top_k": 1}
model_path = "/root/autodl-tmp/glm-4-9b-chat-hf"

# Load dataset and model, tokenizer
ds = load_data('/root/autodl-tmp/data/dataset.json')
model, tokenizer = load_model(model_path)

prompts = [create_prompt_CoT(example['premise'], example['hypothesis']) for example in data]
for i in tqdm(range(0, len(prompts), batch_size), desc="Generating responses"):
    end = min(i + batch_size, len(prompts))
    batch_prompts = prompts[i:end]
    for prompt in batch_prompts:
        # print(prompt)
        tokenized_prompts = tokenize_prompts(prompt, tokenizer)
        tokenized_prompts = tokenized_prompts.to(device)
        generated_responses = generate_responses(tokenized_prompts, model, tokenizer)
        # print(generated_responses)
        label = map_response_to_label(generated_responses)
        predicted_labels.append(label)
        if i % 10 == 0:
            combined_string = ''.join(generated_responses)
            record_responses.append(generated_responses)

true_labels = [example['gold_label'] for example in data]
# print(predicted_labels[:5])
# print(true_labels[:5])

true_labels_filtered = [label for label in true_labels if label != -1]
predicted_labels_filtered = [label for label in predicted_labels if label != -1]

# Calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(true_labels_filtered, predicted_labels_filtered)
precision = precision_score(true_labels_filtered, predicted_labels_filtered, average='binary', zero_division=0)
recall = recall_score(true_labels_filtered, predicted_labels_filtered, average='binary', zero_division=0)
f1 = f1_score(true_labels_filtered, predicted_labels_filtered, average='binary', zero_division=0)

# Calculate confusion matrix
cm = confusion_matrix(true_labels_filtered, predicted_labels_filtered)
tn, fp, fn, tp = cm.ravel()

# Print the results
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}')

# Save the final results with a progress bar
# with open('/root/autodl-tmp/results/hallucination/llama/third_attempt.jsonl', 'w') as outfile:
#     for i in tqdm(range(len(data)), desc="Saving results"):
#         example = data[i]
#         example['predicted_label'] = predicted_labels[i]
#         example['generated_responses'] = generated_responses
#         outfile.write(json.dumps(example) + '\n')