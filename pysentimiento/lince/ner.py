"""
NER for LinCE dataset
"""
import numpy as np
from seqeval.metrics import f1_score
from datasets import load_dataset, load_metric
from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification
from ..preprocessing import preprocess_tweet
from ..training import train_model

metric = load_metric("seqeval")

id2label = [
    'O',
    'B-EVENT',
    'I-EVENT',
    'B-GROUP',
    'I-GROUP',
    'B-LOC',
    'I-LOC',
    'B-ORG',
    'I-ORG',
    'B-OTHER',
    'I-OTHER',
    'B-PER',
    'I-PER',
    'B-PROD',
    'I-PROD',
    'B-TIME',
    'I-TIME',
    'B-TITLE',
    'I-TITLE',
]

label2id = {v:k for k,v in enumerate(id2label)}


def align_labels_with_tokens(labels, word_ids):
    """
    Tomado de https://huggingface.co/course/chapter7/2?fw=pt
    """
    new_labels = []
    current_word = None

    if all(l is None for l in labels):
        # All labels are none => test dataset
        return [None] * len(word_ids)

    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            label = labels[word_id]
            # Same word as previous token
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def compute_metrics(eval_preds):
    """
    Compute metrics for NER
    """
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    ret = {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "macro_f1": f1_score(true_labels, true_predictions, average="macro"),
        "micro_f1": f1_score(true_labels, true_predictions, average="micro"),
        "accuracy": all_metrics["overall_accuracy"],
    }

    for k, v in all_metrics.items():
        if not k.startswith("overall"):
            ret[k + "_f1"] = v["f1"]
            ret[k + "_precision"] = v["precision"]
            ret[k + "_recall"] = v["recall"]

    return ret

def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokenize examples and also realign labels
    """
    tokenized_inputs = tokenizer(
        examples["words"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["labels"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(
            align_labels_with_tokens(labels, word_ids)
        )

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def load_datasets(lang="es", preprocess=True):
    """
    Load NER datasets
    """
    def preprocess_token(t, lang):
        """
        Seguro podemos hacerlo mejor
        """
        return preprocess_tweet(
            t, lang=lang, demoji=False, preprocess_hashtags=False
        )

    lince_ner = load_dataset("lince", "ner_spaeng")

    """
    TODO: None is for test labels which are not available
    """

    lince_ner = lince_ner.map(
        lambda x: {"labels": [label2id.get(x, None) for x in x["ner"]]}
    )

    if preprocess:
        lince_ner = lince_ner.map(
            lambda x: {
                "words": [preprocess_token(word, lang) for word in x["words"]]
            }
        )

    return lince_ner["train"], lince_ner["validation"], lince_ner["test"]

def train(
    base_model, lang, epochs=5,
    metric_for_best_model="micro_f1",
    **kwargs):

    train_dataset, dev_dataset, test_dataset = load_datasets(
        lang=lang
    )

    return train_model(
        base_model,
        train_dataset=train_dataset, dev_dataset=dev_dataset, test_dataset=dev_dataset,
        id2label=id2label, lang=lang, epochs=epochs,
        # Custom stuff for this thing to work
        tokenize_fun=tokenize_and_align_labels,
        auto_class=AutoModelForTokenClassification,
        data_collator_class=DataCollatorForTokenClassification,
        metrics_fun=compute_metrics,
        metric_for_best_model=metric_for_best_model,
        **kwargs
    )