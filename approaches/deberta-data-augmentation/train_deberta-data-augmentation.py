import argparse
import datasets
import numpy
import os
import pandas
import sys
import tempfile
import torch
import transformers
import random
from nltk.corpus import wordnet
import spacy


# GENERIC

values = [ "Self-direction: thought", "Self-direction: action", "Stimulation",  "Hedonism", "Achievement", "Power: dominance", "Power: resources", "Face", "Security: personal", "Security: societal", "Tradition", "Conformity: rules", "Conformity: interpersonal", "Humility", "Benevolence: caring", "Benevolence: dependability", "Universalism: concern", "Universalism: nature", "Universalism: tolerance" ]
labels = sum([[value + " attained", value + " constrained"] for value in values], [])
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

def synonyms_augmentation(sentence, aug_rate=0.1):
    words = sentence.split()
    augmented_sentence = []
    for word in words:
        if random.random() < aug_rate:  # Just augment a word with a probability of aug_rate
            synonyms = [syn.lemmas()[0].name() for syn in wordnet.synsets(word)]
            if synonyms:
                synonym = random.choice(synonyms).replace('_', ' ')
                augmented_sentence.append(synonym)
            else:
                augmented_sentence.append(word)
        else:
            augmented_sentence.append(word)
    return ' '.join(augmented_sentence)

def random_insertion_deletion(sentence, aug_rate=0.1):
    words = sentence.split()
    n = len(words)
    augmented_sentence = []
    i = 0
    while i < n:
        # Probability to delete a word
        if random.random() < aug_rate:
            if random.random() > 0.5 and i < (n - 1):  # Don't delete if it is the last word
                i += 1  # Skip the current word to simulate deletion
        # Insert a new word before the current word with some probability
        if random.random() < aug_rate:
            new_word = random.choice(wordnet.words())
            augmented_sentence.append(new_word)
        # Add the current word to the sentence
        augmented_sentence.append(words[i])
        i += 1
    return ' '.join(augmented_sentence)

def invert_sentence(sentence):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentence)
    # Find the subject and main object
    subject = None
    object = None
    for token in doc:
        if 'subj' in token.dep_:
            subject = token
        if 'obj' in token.dep_:
            object = token
    
    # Reverse subject and object if both are found
    if subject and object:
        inverted_sentence = sentence.replace(subject.text, "OBJ_PLACEHOLDER").replace(object.text, subject.text)
        inverted_sentence = inverted_sentence.replace("OBJ_PLACEHOLDER", object.text)
        return inverted_sentence
    return sentence  # Return the original sentence if it is not possible to reverse

def paraphrase_sentence(sentence, model_name="t5-base", num_beams=5, num_return_sequences=1):
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    model = transformers.T5ForConditionalGeneration.from_pretrained(model_name)

    # Prepare the entry for T5 with the appropriate prefix for paraphrases
    input_text = f"paraphrase: {sentence} </s>"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generar paráfrasis
    paraphrase_results = model.generate(
        inputs, 
        max_length=512, 
        num_beams=num_beams, 
        num_return_sequences=num_return_sequences, 
        early_stopping=True
    )

    # Decodificar los resultados generados a texto
    paraphrases = [tokenizer.decode(result, skip_special_tokens=True) for result in paraphrase_results]
    return paraphrases

def load_dataset(
    directory,
    tokenizer,
    load_labels=True,
    use_synonyms_augmentation=False,
    use_ins_del_augmentation=False,
    use_inversion_augmentation=False,
    use_paraphrases_augmentation=False,
    augmentations=None
):
    sentences_file_path = os.path.join(directory, "sentences.tsv")
    labels_file_path = os.path.join(directory, "labels.tsv")
    
    data_frame = pandas.read_csv(sentences_file_path, encoding="utf-8", sep="\t", header=0)

    # Fix TypeError: TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]
    data_frame['Text'] = data_frame['Text'].fillna('')

    if augmentations:
        for func in augmentations:
            
            data_frame['Text'] = data_frame['Text'].apply(func)

    encoded_sentences = tokenizer(data_frame["Text"].to_list(), truncation=True)

    if load_labels and os.path.isfile(labels_file_path):
        labels_frame = pandas.read_csv(labels_file_path, encoding="utf-8", sep="\t", header=0)
        labels_frame = pandas.merge(data_frame, labels_frame, on=["Text-ID", "Sentence-ID"])
        labels_matrix = numpy.zeros((labels_frame.shape[0], len(labels)))
        for idx, label in enumerate(labels):
            if label in labels_frame.columns:
                labels_matrix[:, idx] = (labels_frame[label] >= 0.5).astype(int)
        encoded_sentences["labels"] = labels_matrix.tolist()

    encoded_sentences = datasets.Dataset.from_dict(encoded_sentences)
    return encoded_sentences, data_frame["Text-ID"].to_list(), data_frame["Sentence-ID"].to_list()


# TRAINING

def train(training_dataset, validation_dataset, pretrained_model, tokenizer, model_name=None, batch_size=8, num_train_epochs=5, learning_rate=2e-5, weight_decay=0.01):
    # https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb
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

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        pretrained_model, problem_type="multi_label_classification",
        num_labels=len(labels), id2label=id2label, label2id=label2id)
    if torch.cuda.is_available():
        print("Using cuda")
        model = model.to('cuda')

    print("TRAINING")
    print("========")
    trainer = transformers.Trainer(model, args,
        train_dataset=training_dataset, eval_dataset=validation_dataset,
        compute_metrics=compute_metrics, tokenizer=tokenizer)

    trainer.train()

    print("\n\nVALIDATION")
    print("==========")
    evaluation = trainer.evaluate()
    for label in labels:
        sys.stdout.write("%-39s %.2f\n" % (label + ":", evaluation["eval_f1-score"][label]))
    sys.stdout.write("\n%-39s %.2f\n" % ("Macro average:", evaluation["eval_marco-avg-f1-score"]))

    return trainer

# COMMAND LINE INTERFACE

cli = argparse.ArgumentParser(prog="DeBERTa")
cli.add_argument("-t", "--training-dataset", required=True)
cli.add_argument("-v", "--validation-dataset")
cli.add_argument("-m", "--model-name")
cli.add_argument("-o", "--model-directory")
args = cli.parse_args()

pretrained_model = "microsoft/deberta-base"
tokenizer = transformers.DeBERTaTokenizer.from_pretrained(pretrained_model)

augment_functions = [
    synonyms_augmentation,
    invert_sentence,
    lambda x: paraphrase_sentence(x)[0],
    random_insertion_deletion
]

training_dataset, training_text_ids, training_sentence_ids = load_dataset(
    args.training_dataset,
    tokenizer,
    use_synonyms_augmentation=True,
    use_ins_del_augmentation=False,
    use_inversion_augmentation=False,
    use_paraphrases_augmentation=False,
    augmentations=augment_functions
)
validation_dataset = training_dataset
if args.validation_dataset != None:
    validation_dataset, validation_text_ids, validation_sentence_ids = load_dataset(args.validation_dataset, tokenizer)
trainer = train(training_dataset, validation_dataset, pretrained_model, tokenizer, model_name = args.model_name)
if args.model_name != None:
    print("\n\nUPLOAD to https://huggingface.co/" + args.model_name + " (using HF_TOKEN environment variable)")
    print("======")
    #trainer.push_to_hub()

if args.model_directory != None:
    print("\n\nSAVE to " + args.model_directory)
    print("======")
    trainer.save_model(args.model_directory)