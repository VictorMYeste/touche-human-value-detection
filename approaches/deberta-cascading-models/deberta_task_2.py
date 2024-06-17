import os
import pandas
import numpy
import torch
import sys
import transformers
from tqdm import tqdm

# GENERIC

values = [ "Self-direction: thought", "Self-direction: action", "Stimulation",  "Hedonism", "Achievement", "Power: dominance", "Power: resources", "Face", "Security: personal", "Security: societal", "Tradition", "Conformity: rules", "Conformity: interpersonal", "Humility", "Benevolence: caring", "Benevolence: dependability", "Universalism: concern", "Universalism: nature", "Universalism: tolerance" ]
id2label = {idx:label for idx, label in enumerate(values)}
label2id = {label:idx for idx, label in enumerate(values)}
labels = sum([[value + " attained", value + " constrained"] for value in values], [])

# SETUP
model_path_1 = "VictorYeste/deberta-based-human-value-detection" # load from huggingface
model_path_2 = "VictorYeste/deberta-based-human-value-stance-detection" # load from huggingface
# model_path_1 = "model_task_1" # load from directory
# model_path_2 = "model_task_2" # load from directory

tokenizer = transformers.AutoTokenizer.from_pretrained(model_path_1)
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path_1)
sigmoid = torch.nn.Sigmoid()

def pipeline_1(text):
    # Source: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb
    """ Predicts the value probabilities (attained and constrained) for each sentence """
    encoding = tokenizer(text, return_tensors="pt")
    encoding = {k: v for k,v in encoding.items()}
    outputs = model(**encoding)
    logits = outputs.logits
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    probs = probs.detach().cpu().numpy().tolist()
    predicted_labels = [id2label[idx] for idx in range(len(probs))]
    predicted_scores = [probs[idx] for idx in range(len(probs))]
    return predicted_labels, predicted_scores

pipeline_2 = transformers.pipeline("text-classification", model=model_path_2, tokenizer=model_path_2, top_k=None)


# PREDICTION

# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb
def predict(text):
    """ Predicts the value probabilities (attained and constrained) for each sentence """

    final_results = []
    for sentence_text in text:
        model_1_labels, model_1_scores = pipeline_1(sentence_text)
        # Prediction using model two for the human value with largest confidence score, to be used as default:
        pred_dict = {}
        for hvalue in values:
            input_for_model_2 = f"{sentence_text} {hvalue}"
            model_2_results = pipeline_2(input_for_model_2, truncation=True)
            predid = model_1_labels.index(hvalue)
            for x in model_2_results[0]:
                if model_1_scores[predid] >= 0.5:
                    pred_dict[hvalue + " " + x["label"]] = x["score"]
                else:
                    pred_dict[hvalue + " " + x["label"]] = (x["score"] * (model_1_scores[predid] / 2.0))          
        
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
