import pandas as pd
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments,DataCollatorForSeq2Seq
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("C:\\vitAI\\vitAI_BART_model", local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained("C:\\vitAI\\vitAI_BART_model", local_files_only=True).to("cuda")

# Load the dataset
dataset = load_dataset('csv', data_files={'test': 'C:\\vitAI\\cleaned_data\\generated_test_data\\test_data_1.csv'},column_names=['dataset','id','dialogue','note'])

# Adjust generation parameters for longer outputs
gen_kwargs = {
    "max_length": 256,  # Increase as needed
    "min_length": 150,  # Adjust based on desired minimum summary length
    "length_penalty": 2.0,  # Encourages longer sentences
    "num_beams": 4,  # Increase for more diverse candidates
    "early_stopping": True,
}

# Metrics
rouge = load_metric('rouge')
meteor = load_metric('meteor')
bleu = load_metric('bleu')
accuracy_metric = load_metric('accuracy')
f1_metric = load_metric('f1')

# Prepare results container
results = []

def compute_metrics(predictions, references):
    rouge_score = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    meteor_score = meteor.compute(predictions=predictions, references=references)
    bleu_score = bleu.compute(predictions=[predictions], references=[[references]])
    accuracy_score = accuracy_metric.compute(predictions=predictions, references=references)
    f1_score = f1_metric.compute(predictions=predictions, references=references, average='macro')
    
    return {
        "ROUGE-1": rouge_score['rouge1'].mid.fmeasure,
        "ROUGE-2": rouge_score['rouge2'].mid.fmeasure,
        "ROUGE-L": rouge_score['rougeL'].mid.fmeasure,
        "METEOR": meteor_score['score'],
        "BLEU": bleu_score['score'],
        "Accuracy": accuracy_score['accuracy'],
        "F1": f1_score['f1']
    }

# Evaluate the model
model.eval()
for example in dataset['test']:
    inputs = tokenizer(example['dialogue'], return_tensors="pt", padding=True, truncation=True, max_length=1024)
   
    
    outputs = model.generate(inputs['input_ids'].to("cuda"),attention_mask=inputs["attention_mask"], **gen_kwargs)
    
    pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    target_text = example['note']
    
    metrics = compute_metrics([pred_text], [target_text])
    results.append(metrics)

# Save results to a CSV file
df = pd.DataFrame(results)
df.to_csv("C:\\vitAI\\evaluation_metrics\\model_evaluation_metrics.csv", index=False)
