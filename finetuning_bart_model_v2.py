from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, load_metric
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Initialize the tokenizer and model
device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained("lidiya/bart-large-xsum-samsum")
model = AutoModelForSeq2SeqLM.from_pretrained("lidiya/bart-large-xsum-samsum")
model.to(device)
collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Load the dataset
dataset = load_dataset('csv', data_files={
    'train': ['C:\\vitAI\\cleaned_data\\cleaned_aci-bench\\train.csv'], 
    'test': ['C:\\vitAI\\cleaned_data\\cleaned_aci-bench\\test.csv'],
    'validate': ['C:\\vitAI\\cleaned_data\\cleaned_aci-bench\\valid.csv']
}, column_names=['dataset','id','dialogue','note'])

# Define preprocessing function
def preprocess_function(examples):
    inputs = examples["dialogue"]
    targets = examples["note"]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Load metrics
rouge = load_metric('rouge')
bleu = load_metric('bleu')
meteor = load_metric('meteor')
accuracy_metric = load_metric('accuracy')
f1_metric = load_metric('f1')

def compute_metrics(pred):
    predictions, labels = pred
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    rouge_score = rouge.compute(predictions=predictions, references=labels)
    bleu_score = bleu.compute(predictions=[predictions], references=[[labels]])
    meteor_score = meteor.compute(predictions=predictions, references=labels)
    accuracy_score = accuracy_metric.compute(predictions=predictions, references=labels)
    f1_score = f1_metric.compute(predictions=predictions, references=labels, average='macro')

    return {
        "rouge": rouge_score['mid']['fmeasure'],
        "bleu": bleu_score['score'],
        "meteor": meteor_score['score'],
        "accuracy": accuracy_score['accuracy'],
        "f1": f1_score['f1']
    }

# Set training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",                    # path for saving output files
    evaluation_strategy="epoch",                          
    logging_dir='./logs',                      # directory for storing logs
    logging_steps=100,
    per_device_train_batch_size=2,             # batch size for training
    per_device_eval_batch_size=2,              # batch size for evaluation
    weight_decay=0.01,                            # no weight decay
    save_total_limit=3,
    num_train_epochs=3,                       # total number of training epochs
    predict_with_generate=True,
    generation_max_length=256,                 # maximum length of the output sequences
    learning_rate=1e-5,                        # learning rate
    fp16=True,                                 # use mixed precision
    
)

# Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validate'],
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# save the model
directory = "C:\\vitAI\\vitAI_BART_model_2"
trainer.save_model(directory)
tokenizer.save_vocabulary(directory)
model.save_pretrained(directory)
print('All files for model saved')

# Tensorboard for visualization
writer = SummaryWriter()
writer.add_graph(model, torch.randn(1, 1024).to(torch.int64))
writer.close()

# test the model on a conversation
input_text = """[doctor] hi , andrew . how are you ? [patient] hey , good to see you . [doctor] i'm doing well , i'm doing well . [patient] good . [doctor] so , i know the nurse told you about dax . i'd like to tell dax a little bit about you . [patient] sure . [doctor] uh , so , andrew is a 59-year-old male with a past medical history , significant for depression , type two diabetes , and hypertension who presents today with an upper respiratory infection . so , andrew , what's going on ? [patient] yeah . we were doing a bit of work out in the yard in the last week or so and i started to feel really tired , was short of breath . um , we- we're not wearing masks as much at the end of the summer and i think i caught my first cold and i think it just got worse . [doctor] okay . all right . um , now , have you had your covid vaccines ? [patient] yeah , both . [doctor] okay . all right . and , um , do you have any history of any seasonal allergies at all ? [patient] none whatsoever . [doctor] okay . all right . and when you say you're having some shortness of breath , did you feel short of breath walking around or at rest ? [patient] uh , usually , it was lifting or carrying something . we were doing some landscaping , so i was carrying some heavy bags of soil and i , i got really winded . it really surprised me . [doctor] okay . and are you coughing up anything ? [patient] not yet , but i feel like that's next . [doctor] okay . and fevers ? [patient] uh , i felt a little warm , but i , i just thought it was because i was exerting myself . [doctor] okay . all right . and any other symptoms like muscle aches , joint pain , fatigue ? [patient] my elbows hurt quite a bit and my knees were pretty tired . l- like i said , i really felt some tension around my knees , but , uh , i think that was a lot to do with , uh , lifting the bags . [doctor] okay . all right . um , so , you know , how about , how are you doing in terms of your other medical problems , like your depression ? how are you doing with that ? i know we've , you know , talked about not putting you on medication for it because you're on medication for other things . what's going on ? [patient] i- it's been kind of a crazy year and a half . i was a little concerned about that but , for the most part , i've been , been doing well with it . my , my wife got me into barre classes , to help me relax and i think it's working . [doctor] okay . all right , great . and , and in terms of your diabetes , how are you doing watching your , your diet and your sugar intake ? [patient] uh , i've been monitoring my sugar levels while i am going to work during the week . uh , not so , uh , if its saturday or sunday i usually don't remember . uh , the diet's been pretty good for the most part , except for , you know , some house parties and things like that . but , uh , been good for the most part . [doctor] okay and have they been elevated at all since this episode of your- [patient] no . [doctor] okay . and then , how , lastly , for your high blood pressure , have you been monitoring your blood pressures at home ? did you buy the cuff like i suggested ? [patient] uh , same thing . during the while i'm going to work, i'm regular about monitoring it, but if its a saturday or sunday, not so much . but , uh , it's , it's been under control . [doctor] but you're taking your medication ? [patient] yes . [doctor] okay . all right . well , you know , i know that , you know , you've endorsed , you know , the shortness of breath and some joint pain . um , how about any other symptoms ? nausea or vomiting ? diarrhea ? [patient] no . [doctor] anything like that ? [patient] no . [doctor] okay . all right . well , i wan na go ahead and do a quick physical exam , all right ? hey , dragon , show me the vital signs . so , your vital signs here in the office look quite good . [patient] mm-hmm . [doctor] you know , everything's looking normal , you do n't have a fever , which is really good . um , i'm just gon na go ahead and listen to your heart and your lungs and , kind of , i'll let you know what i hear , okay ? [patient] sure . [doctor] okay . so , on your physical exam , you know , your heart sounds nice and strong . your lungs , you do have scattered ronchi bilaterally on your lung exam . uh , it clears with cough . um , i do notice a little bit of , um , some edema of your lower extremities and you do have some pain to palpation of your elbows bilaterally . um , so , let's go ahead , i want to look at some of your results , okay ? [patient] mm-hmm . [doctor] hey , dragon . show me the chest x-ray . [doctor] so , i reviewed the results of your chest x-ray and everything looks good . there's no airspace disease , there's no pneumonia , so that's all very , very good , okay ? [patient] good . [doctor] hey , dragon . show me the diabetic labs . [doctor] and here , looking at your diabetic labs , you know , your hemoglobin a1c is a little elevated at eight . [patient] mm-hmm . [doctor] i'd like to see that a little bit better , around six or seven , if possible . [patient] mm-hmm . [doctor] um , so let's talk a little bit about my assessment and my plan for you . [patient] mm-hmm . [doctor] so , for your first problem , this upper respiratory infection , i believe you , you have a viral syndrome , okay ? we'll go ahead and we'll send a covid test , just to make sure that you do n't have covid . [patient] mm-hmm . [doctor] uh , but overall , i think that , um , you know , this will resolve in a couple of days . i do n't think you have covid , you do n't have any exposures , that type of thing . [patient] mm-hmm . [doctor] so , i think that this will improve . i'll give you some robitussin for your cough and i would encourage you take some ibuprofen , tylenol for any fever , okay ? [patient] you got it . [doctor] for your next problem , your depression , you know , it sounds like you're doing well with that , but again , i'm happy to start on a med- , a medical regiment or ... [patient] mm-hmm . [doctor] . refer you to psychotherapy , if you think that that would be helpful . [patient] mm-hmm . [doctor] would you like that ? [patient] u- u- um , maybe not necessarily . maybe in a , uh , few months we'll check on that . [doctor] okay . all right . [doctor] for your third problem , your type two diabetes , i want to go ahead and increase your metformin to 1000 milligrams , twice daily . [patient] mm-hmm . [doctor] and i'm gon na get an- another hemoglobin a1c in four months , okay ? [patient] okay , sure . [doctor] hey , dragon . order a hemoglobin a1c . [doctor] and lastly , for your high blood pressure , it looks like you're doing a really good job managing that . i want to go ahead and continue you on the , um , lisinopril , 20 milligrams a day . [patient] mm-hmm . [doctor] and i'm gon na go ahead and order a lipid panel , okay ? [patient] sure . [doctor] do you need a refill of the lisinopril ? [patient] actually , i do . [doctor] okay . hey , dragon . order lisinopril , 20 milligrams daily . [doctor] so , the nurse will be in , she'll help you , uh , make a follow-up appointment with me . i want to see you again in about four months . [patient] okay . [doctor] let me know if your symptoms worsen and we can talk more about it , okay ? [patient] you got it . [doctor] all right . hey , dragon . finalize the note ."""
inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True, padding="max_length")
inputs = inputs.to("cuda:0")
# Adjust generation parameters for longer outputs
gen_kwargs = {
    "max_length": 256,  # Increase as needed
    "min_length": 150,  # Adjust based on desired minimum summary length
    "length_penalty": 2.0,  # Encourages longer sentences
    "num_beams": 4,  # Increase for more diverse candidates
    "early_stopping": True,
}

# Generate summary
summary_ids = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], **gen_kwargs)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(summary)
