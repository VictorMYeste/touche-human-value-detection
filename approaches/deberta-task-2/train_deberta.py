import argparse
import datasets
import numpy
import os
import pandas
import sys
import tempfile
import torch
import transformers

# from huggingface_hub import notebook_login
# notebook_login()

# GENERIC

values = [ "Self-direction: thought", "Self-direction: action", "Stimulation",  "Hedonism", "Achievement", "Power: dominance", "Power: resources", "Face", "Security: personal", "Security: societal", "Tradition", "Conformity: rules", "Conformity: interpersonal", "Humility", "Benevolence: caring", "Benevolence: dependability", "Universalism: concern", "Universalism: nature", "Universalism: tolerance" ]
labels_subtask1 = values[:]
id2label_subtask1 = {idx:label for idx, label in enumerate(labels_subtask1)}
label2id_subtask1 = {label:idx for idx, label in enumerate(labels_subtask1)} 
labels = sum([[value + " attained", value + " constrained"] for value in values], [])
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)} 

def load_dataset(directory, tokenizer, load_labels=True, sample_rate=1.0):
    sentences_file_path = os.path.join(directory, "sentences.tsv")
    labels_file_path = os.path.join(directory, "labels.tsv")
    
    data_frame = pandas.read_csv(sentences_file_path, encoding="utf-8", sep="\t", header=0)

    if sample_rate < 1.0:
        data_frame = data_frame.sample(frac=sample_rate, random_state=42).reset_index(drop=True)

    # Fix TypeError: TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]
    data_frame['Text'] = data_frame['Text'].fillna('')

    encoded_sentences = tokenizer(data_frame["Text"].to_list(), truncation=True)

    if load_labels and os.path.isfile(labels_file_path):
        labels_frame = pandas.read_csv(labels_file_path, encoding="utf-8", sep="\t", header=0)
        # Sub-task 1: Convert labels with attained and constrained to only presence
        for label in labels_subtask1:
            attained_col = label + " attained"
            constrained_col = label + " constrained"
            labels_frame[label] = ((labels_frame[attained_col] > 0.0) | (labels_frame[constrained_col] > 0.0)).astype(float)
        # End sub-task-1
        labels_frame = pandas.merge(data_frame, labels_frame, on=["Text-ID", "Sentence-ID"])
        labels_matrix = numpy.zeros((labels_frame.shape[0], len(labels_subtask1)))
        for idx, label in enumerate(labels_subtask1):
            if label in labels_frame.columns:
                labels_matrix[:, idx] = (labels_frame[label] >= 0.5).astype(int)
        encoded_sentences["labels"] = labels_matrix.tolist()

    encoded_sentences = datasets.Dataset.from_dict(encoded_sentences)
    return encoded_sentences, data_frame["Text-ID"].to_list(), data_frame["Sentence-ID"].to_list()


# TRAINING

def train(training_dataset, validation_dataset, subtask, pretrained_model, tokenizer, model_name=None, batch_size=8, num_train_epochs=5, learning_rate=2e-5, weight_decay=0.01):
    # https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb
    if subtask == 1:
        def compute_metrics(eval_prediction):
            prediction_scores, label_scores = eval_prediction
            predictions = prediction_scores >= 0.0 # sigmoid
            labels_subtask1 = label_scores >= 0.5

            f1_scores = {}
            for i in range(predictions.shape[1]):
                predicted = predictions[:,i].sum()
                true = labels_subtask1[:,i].sum()
                true_positives = numpy.logical_and(predictions[:,i], labels_subtask1[:,i]).sum()
                precision = 0 if predicted == 0 else true_positives / predicted
                recall = 0 if true == 0 else true_positives / true
                f1_scores[id2label_subtask1[i]] = round(0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall), 2)
            macro_average_f1_score = round(numpy.mean(list(f1_scores.values())), 2)

            return {'f1-score': f1_scores, 'marco-avg-f1-score': macro_average_f1_score}
    else:
        def compute_metrics(eval_prediction):
            prediction_scores, label_scores = eval_prediction
            predictions = prediction_scores >= 0.0 # sigmoid
            labels = label_scores >= 0.5

            f1_scores = {}
            for i in range(predictions.shape[1]):
                predicted = predictions[:,i].sum()
                true = labels[:,i].sum()
                true_positives = numpy.logical_and(predictions[:,i], labels[:,i]).sum()
                precision = 0 if predicted == 0 else true_positives / predicted
                recall = 0 if true == 0 else true_positives / true
                f1_scores[id2label[i]] = round(0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall), 2)
            macro_average_f1_score = round(numpy.mean(list(f1_scores.values())), 2)

            return {'f1-score': f1_scores, 'marco-avg-f1-score': macro_average_f1_score}

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
        metric_for_best_model='marco-avg-f1-score'
    )

    if subtask == 1:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model, problem_type="multi_label_classification",
            num_labels=len(labels_subtask1), id2label=id2label_subtask1, label2id=label2id_subtask1)
    else:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model, problem_type="multi_label_classification",
            num_labels=len(labels), id2label=id2label, label2id=label2id)
    if torch.cuda.is_available():
        print("Using cuda")
        model = model.to('cuda')

    if subtask == 1:
        print("SUBTASK 1")
    else:
        print("SUBTASK 2")

    print("TRAINING")
    print("========")
    def custom_collate_fn(batch):
        collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
        batch = collator(batch)
        return batch
    
    trainer = transformers.Trainer(model, args,
        train_dataset=training_dataset, eval_dataset=validation_dataset,
        compute_metrics=compute_metrics, tokenizer=tokenizer,
        data_collator=custom_collate_fn)

    trainer.train()

    print("\n\nVALIDATION")
    print("==========")
    evaluation = trainer.evaluate()
    if subtask == 1:
        for label in labels_subtask1:
            sys.stdout.write("%-39s %.2f\n" % (label + ":", evaluation["eval_f1-score"][label]))
        sys.stdout.write("\n%-39s %.2f\n" % ("Macro average:", evaluation["eval_marco-avg-f1-score"]))
    else:
        for label in labels:
            sys.stdout.write("%-39s %.2f\n" % (label + ":", evaluation["eval_f1-score"][label]))
        sys.stdout.write("\n%-39s %.2f\n" % ("Macro average:", evaluation["eval_marco-avg-f1-score"]))

    return trainer, model

def filter_data_based_on_first_model(model, training_dataset, validation_dataset, threshold=0.5):
    # Set the model in evaluation mode
    model.eval()

    # Use the Trainer to get predictions from datasets
    trainer = transformers.Trainer(model=model)

    # Get predictions for the training dataset
    train_results = trainer.predict(training_dataset)
    # Extract scores (ensure softmax is applied if model outputs logits)
    train_scores = torch.nn.functional.softmax(torch.tensor(train_results.predictions), dim=-1)
    # Apply a threshold to determine which inputs pass to the next model
    train_indices = (train_scores.max(axis=1).values > threshold).nonzero(as_tuple=True)[0].numpy()
    # Filter original training dataset to retain only selected entries
    filtered_train_dataset = training_dataset.select(train_indices.tolist())

    # Repeat the process for validation dataset
    validation_results = trainer.predict(validation_dataset)
    validation_scores = torch.nn.functional.softmax(torch.tensor(validation_results.predictions), dim=-1)
    validation_indices = (validation_scores.max(axis=1).values > threshold).nonzero(as_tuple=True)[0].numpy()
    filtered_validation_dataset = validation_dataset.select(validation_indices.tolist())

    # Filter original datasets to retain only selected entries
    filtered_train_dataset = training_dataset.select(train_indices)
    filtered_validation_dataset = validation_dataset.select(validation_indices)

    return filtered_train_dataset, filtered_validation_dataset

def push_model_to_hub(trainer, model_name, private=True):
    if model_name:
        trainer.push_to_hub(
            repo_name=model_name,
            use_auth_token=True,
            private=private
        )
        print(f"Model {model_name} successfully loaded into Hugging Face Hub.")
    else:
        print("Model name is None or empty, not pushing to hub.")

# COMMAND LINE INTERFACE

cli = argparse.ArgumentParser(prog="DeBERTa")
cli.add_argument("-t", "--training-dataset", required=True)
cli.add_argument("-v", "--validation-dataset")
cli.add_argument("-m", "--model-name")
cli.add_argument("-o", "--model-directory")
args = cli.parse_args()

pretrained_model = "microsoft/deberta-base"
tokenizer = transformers.DebertaTokenizer.from_pretrained(pretrained_model)

training_dataset, training_text_ids, training_sentence_ids = load_dataset(args.training_dataset, tokenizer, sample_rate=0.01)
validation_dataset = training_dataset
if args.validation_dataset != None:
    validation_dataset, validation_text_ids, validation_sentence_ids = load_dataset(args.validation_dataset, tokenizer, sample_rate=0.01)

# Subtask 1: Train
subtask = 1
trainer_subtask1, model_subtask1 = train(training_dataset, validation_dataset, subtask, pretrained_model, tokenizer, model_name = args.model_name)

# Subtask 2: Filter
filtered_training_data, filtered_validation_data = filter_data_based_on_first_model(model_subtask1, training_dataset, validation_dataset)
# Subtask 2: Train
subtask = 2
trainer_subtask2, _ = train(filtered_training_data, filtered_validation_data, subtask, pretrained_model, tokenizer, model_name=args.model_name)

if args.model_name != None:
    print("\n\nUPLOAD to https://huggingface.co/" + args.model_name + " (using HF_TOKEN environment variable)")
    print("======")
    # push_model_to_hub(trainer_subtask1, args.model_name + "_Subtask_1", private=True)
    # push_model_to_hub(trainer_subtask2, args.model_name + "_Subtask_2", private=True)

if args.model_directory != None:
    print("\n\nSAVE to " + args.model_directory)
    print("======")
    trainer_subtask1.save_model(args.model_directory + "/subtask1")
    trainer_subtask2.save_model(args.model_directory + "/subtask2")