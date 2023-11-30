from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import json

# Step 1: Load and Preprocess the Data
with open('assets//edu.json', 'r') as f:
    data = json.load(f)

with open('assets//entries.json', 'r') as f:
    data.append(json.load(f))

corpus = [item['text'] for item in data]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

question = "What is the capital of France?"
for text in corpus:
    inputs = tokenizer(question, text, return_tensors='pt', truncation=True)

    # Step 4: Inference with BERT
    outputs = model(**inputs)
    
    # Extracting start and end positions of the answer
    start_idx = torch.argmax(outputs['start_logits'])
    end_idx = torch.argmax(outputs['end_logits']) + 1
    
    # Extracting the answer
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_idx:end_idx]))

    print(f"Question: {question}")
    print(f"Answer: {answer}")
