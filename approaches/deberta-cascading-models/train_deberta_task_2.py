import argparse
import datasets
import numpy
import os
import pandas
import sys
import tempfile
import torch
import transformers

import evaluate
f1 = evaluate.load("f1")
accuracy = evaluate.load("accuracy")

# GENERIC
values = [ "Self-direction: thought", "Self-direction: action", "Stimulation",  "Hedonism", "Achievement", "Power: dominance", "Power: resources", "Face", "Security: personal", "Security: societal", "Tradition", "Conformity: rules", "Conformity: interpersonal", "Humility", "Benevolence: caring", "Benevolence: dependability", "Universalism: concern", "Universalism: nature", "Universalism: tolerance" ]
labels = [ "attained", "constrained" ]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)} 

def load_dataset(directory):
    sentences_file_path = os.path.join(directory, "sentences.tsv")
    labels_file_path = os.path.join(directory, "labels.tsv")

    data_frame = pandas.read_csv(
        sentences_file_path, encoding="utf-8", sep="\t", header=0
    )

    sentences_df = pandas.read_csv(sentences_file_path, encoding="utf-8", sep="\t", header=0)
    labels_df = pandas.read_csv(labels_file_path, encoding="utf-8", sep="\t", header=0)

    # Fix TypeError: TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]
    sentences_df["Text"] = sentences_df["Text"].fillna("")
    sentences_df = pandas.merge(sentences_df, labels_df, on=["Text-ID", "Sentence-ID"])

    # Crear pares premise-hypothesis
    data = []
    for _, row in sentences_df.iterrows():
        sentence = row['Text']
        for column in values:
            attn = row[column + " attained"]
            cnst = row[column + " constrained"]
            if (attn + cnst) >= 0.5:
                if attn > cnst:
                    data.append({'premise': sentence, 'hypothesis': column, 'label': 'attained'})
                elif cnst > attn:
                    data.append({'premise': sentence, 'hypothesis': column, 'label': 'constrained'})

    data = pandas.DataFrame(data)
    
    data['label'] = data['label'].map({'attained': 0, 'constrained': 1})

    encoded_sentences = datasets.Dataset.from_pandas(data)

    return encoded_sentences

# TRAINING

def train(training_dataset, validation_dataset, pretrained_model, tokenizer, model_name=None, batch_size=8, num_train_epochs=5, learning_rate=2e-5, weight_decay=0.01):
    # https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb
    def compute_metrics(eval_prediction):
        prediction_scores, label_scores = eval_prediction
        predictions = numpy.argmax(prediction_scores, axis = 1)
        f1_score_macro = f1.compute(predictions=predictions, references=label_scores, average="macro")
        f1_score_micro = f1.compute(predictions=predictions, references=label_scores, average="micro")
        accuracy_score = accuracy.compute(predictions=predictions, references=label_scores)

        return {"f1 macro": f1_score_macro, "f1 micro": f1_score_micro, "accuracy": accuracy_score}


    output_dir = tempfile.TemporaryDirectory()
    args = transformers.TrainingArguments(
        output_dir=output_dir.name,
        save_strategy="epoch",
        hub_model_id=model_name,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
    )

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        pretrained_model,
        num_labels=len(labels), id2label=id2label, label2id=label2id)
    if torch.cuda.is_available():
        print("Using cuda")
        model = model.to('cuda')

    print("TRAINING")
    print("========")
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    trainer = transformers.Trainer(model, args,
        train_dataset=training_dataset, eval_dataset=validation_dataset,
        compute_metrics=compute_metrics, tokenizer=tokenizer, data_collator=data_collator
        )

    trainer.train()

    print("\n\nVALIDATION")
    print("==========")
    evaluation = trainer.evaluate()
    print(evaluation)
    return trainer

# COMMAND LINE INTERFACE

cli = argparse.ArgumentParser(prog="DeBERTa")
cli.add_argument("-t", "--training-dataset", required=True)
cli.add_argument("-v", "--validation-dataset")
cli.add_argument("-m", "--model-name")
cli.add_argument("-o", "--model-directory")
args = cli.parse_args()

pretrained_model = "microsoft/deberta-base"
tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model)

def encode(inference):
    return tokenizer(inference['premise'], inference['hypothesis'], truncation=True)

training_dataset = load_dataset(args.training_dataset)
training_dataset = training_dataset.map(encode, batched=True)

validation_dataset = training_dataset
if args.validation_dataset != None:
    validation_dataset = load_dataset(args.validation_dataset)
validation_dataset = validation_dataset.map(encode, batched=True)

trainer = train(training_dataset, validation_dataset, pretrained_model, tokenizer, model_name = args.model_name)

if args.model_name != None:
    print("\n\nUPLOAD to https://huggingface.co/" + args.model_name + " (using HF_TOKEN environment variable)")
    print("======")
    #trainer.push_to_hub()

if args.model_directory != None:
    print("\n\nSAVE to " + args.model_directory)
    print("======")
    trainer.save_model(args.model_directory)