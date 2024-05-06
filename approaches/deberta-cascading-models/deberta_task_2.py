import os
import datasets
import pandas
import numpy
import torch
import sys
import transformers
import tempfile
from tqdm import tqdm

# GENERIC

values = [ "Self-direction: thought", "Self-direction: action", "Stimulation",  "Hedonism", "Achievement", "Power: dominance", "Power: resources", "Face", "Security: personal", "Security: societal", "Tradition", "Conformity: rules", "Conformity: interpersonal", "Humility", "Benevolence: caring", "Benevolence: dependability", "Universalism: concern", "Universalism: nature", "Universalism: tolerance" ]
labels = sum([[value + " attained", value + " constrained"] for value in values], [])
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)} 

# SETUP

model_path_1 = "model_task_1"
tokenizer_1 = transformers.AutoTokenizer.from_pretrained(model_path_1)
model_1 = transformers.AutoModelForSequenceClassification.from_pretrained(model_path_1)
pipeline_1 = transformers.pipeline("text-classification", model=model_1, tokenizer=tokenizer_1, top_k=None)

model_path_2 = "model_task_2"
tokenizer_2 = transformers.AutoTokenizer.from_pretrained(model_path_2)
model_2 = transformers.AutoModelForSequenceClassification.from_pretrained(model_path_2)
pipeline_2 = transformers.pipeline("text-classification", model=model_2, tokenizer=tokenizer_2)

# PREDICTION

# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb
def predict(text):
    """ Predicts the value probabilities (attained and constrained) for each sentence """
    # "text" contains all sentences (plain strings) of a single text in order (same Text-ID in the input file)

    # Apply Model 1
    model_1_results = pipeline_1(text, truncation=True)

    final_results = []

    for result_1 in tqdm(model_1_results, desc="Labeling sentences", unit="sentence"):
        for sentence in result_1:
            pred_dict = {}
            value = sentence['label']

            # Initialize predictions as 0.0
            for label in labels:
                pred_dict[label] = 0.0

            # If this value is present in the text
            if sentence['score'] > 0.5:

                # Apply Model 2
                input_for_model_2 = f"{sentence} {value}"
                model_2_results = pipeline_2(input_for_model_2, truncation=True)

                # print(model_2_results)

                # Update predictions with the Model 2 results
                for result_2 in model_2_results:
                    if result_2['score'] >= 0.6:
                        pred_dict[value + " " + result_2['label']] = 1.0
                    elif result_2['score'] >= 0.4:
                        pred_dict[value + " attained"] = 0.5
                        pred_dict[value + " constrained"] = 0.5

                final_results.append(pred_dict)

    return final_results

# EXECUTION

def label(instances):
    """ Predicts the label probabilities for each instance and adds them to it """
    text = [instance["Text"] for instance in instances]
    return [{
            "Text-ID": instance["Text-ID"],
            "Sentence-ID": instance["Sentence-ID"],
            **labels
        } for instance, labels in zip(instances, predict(text))]

def writeRun(labeled_instances, output_dir):
    """ Writes all (labeled) instances to the predictions.tsv in the output directory """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "predictions.tsv")
    pandas.DataFrame.from_dict(labeled_instances).to_csv(output_file, header=True, index=False, sep='\t')

# code not executed by tira-run-inference-server (which directly calls 'predict(text)')
if "TIRA_INFERENCE_SERVER" not in os.environ:
    dataset_dir = sys.argv[1]
    output_dir = sys.argv[2]
    labeled_instances = []
    input_file = os.path.join(dataset_dir, "sentences.tsv")
    texts_df = pandas.read_csv(input_file, sep='\t', header=0, index_col=None).groupby("Text-ID")
    for text_instances in tqdm(texts_df, desc="Labeling Texts", unit="text"):
        # label the instances of each text separately
        labeled_instances.extend(label(text_instances[1].sort_values("Sentence-ID").to_dict("records")))
    writeRun(labeled_instances, output_dir)