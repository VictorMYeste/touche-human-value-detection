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
id2label = {idx:label for idx, label in enumerate(values)}
label2id = {label:idx for idx, label in enumerate(values)}
labels = sum([[value + " attained", value + " constrained"] for value in values], [])

# SETUP
model_path_1 = "model_task_1"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path_1)  # load from directory
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path_1)  # load from directory
sigmoid = torch.nn.Sigmoid()

def pipeline_1(text):
    # Source: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb
    """ Predicts the value probabilities (attained and constrained) for each sentence """
    # "text" contains all sentences (plain strings) of a single text in order (same Text-ID in the input file)
    encoding = tokenizer(text, return_tensors="pt")
    encoding = {k: v for k,v in encoding.items()}
    outputs = model(**encoding)
    logits = outputs.logits
    # apply sigmoid + threshold
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = numpy.zeros(probs.shape)
    # predictions[numpy.where(probs >= 0.5)] = 1
    predicted_labels = [id2label[idx] for idx, label in enumerate(predictions)]
    predicted_scores = [probs[idx].item() for idx, label in enumerate(predictions)]
    return predicted_labels, predicted_scores

model_path_2 = "models/model_task_2"
pipeline_2 = transformers.pipeline("text-classification", model=model_path_2, tokenizer=model_path_2, top_k=None)


# PREDICTION

# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb
def predict(text):
    """ Predicts the value probabilities (attained and constrained) for each sentence """
    # "text" contains all sentences (plain strings) of a single text in order (same Text-ID in the input file)

    final_results = []
    for sentence_text in text:
        model_1_labels, model_1_scores = pipeline_1(sentence_text)
        
        # pred_dict = {}
        # for hvalue in values:
        #     if hvalue in model_1_results:
        #         input_for_model_2 = f"{sentence_text} {hvalue}"
        #         model_2_results = pipeline_2(input_for_model_2, truncation=True)
        #         if model_2_results[0]["score"] >= 0.6:
        #             if model_2_results[0]['label'] == "attained":
        #                 pred_dict[hvalue + " attained"] = 1.0
        #                 pred_dict[hvalue + " constrained"] = 0.0
        #             else:
        #                 pred_dict[hvalue + " attained"] = 0.0
        #                 pred_dict[hvalue + " constrained"] = 1.0
        #         elif model_2_results[0]['score'] >= 0.4:
        #             pred_dict[hvalue + " attained"] = 0.5
        #             pred_dict[hvalue + " constrained"] = 0.5
        #     else:
        #         pred_dict[hvalue + " attained"] = 0.0
        #         pred_dict[hvalue + " constrained"] = 0.0

        pred_dict = {}
        for hvalue in values:
            predid = model_1_labels.index(hvalue)
            if model_1_scores[predid] >= 0.5:
                input_for_model_2 = f"{sentence_text} {hvalue}"
                model_2_results = pipeline_2(input_for_model_2, truncation=True)
                for x in model_2_results[0]:
                    pred_dict[hvalue + " " + x["label"]] = x["score"]
            else:
                pred_dict[hvalue + " attained"] = model_1_scores[predid]
                pred_dict[hvalue + " constrained"] = model_1_scores[predid]
        
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