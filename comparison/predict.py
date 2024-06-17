import argparse
import random
import pandas as pd
import tweetnlp
import stanza
import logging
from tqdm.auto import tqdm
from pysentimiento import create_analyzer
from textblob import TextBlob
from datasets import load_dataset, ClassLabel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report
from flair.data import Sentence
from flair.nn import Classifier


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sent_eval():
    """
    Loads the SentEval-CR dataset
    """
    sent_eval = load_dataset("SetFit/SentEval-CR")["test"]
    sent_eval = sent_eval.rename_column("text", "sentence")
    sent_eval = sent_eval.cast_column(
        "label", ClassLabel(2, names=["negative", "positive"])
    )
    return sent_eval


def load_sentiment140():
    """
    Loads the Sentiment140 dataset
    """

    sent140 = load_dataset(
        "stanfordnlp/sentiment140", trust_remote_code=True, split="test"
    )
    sent140 = sent140.map(lambda x: {"sentiment": x["sentiment"] / 2})
    sent140 = sent140.rename_column("text", "sentence")
    sent140 = sent140.rename_column("sentiment", "label")

    sent140 = sent140.cast_column(
        "label", ClassLabel(3, names=["negative", "neutral", "positive"])
    )

    return sent140


def load_mteb(lang):

    mteb = load_dataset("mteb/tweet_sentiment_multilingual", lang, split="test")
    mteb = mteb.rename_column("text", "sentence")
    mteb = mteb.cast_column(
        "label", ClassLabel(3, names=["negative", "neutral", "positive"])
    )

    return mteb


def load_amazon(lang):
    """
    Load the Amazon Reviews dataset
    """
    amazon = load_dataset(f"SetFit/amazon_reviews_multi_{lang}", split="test")

    def convert_label(ex):
        if ex["label"] <= 1:
            return 0
        elif ex["label"] >= 3:
            return 2
        else:
            return 1

    amazon = amazon.map(lambda ex: {"label": convert_label(ex)})
    amazon = amazon.cast_column(
        "label", ClassLabel(3, names=["negative", "neutral", "positive"])
    )
    amazon = amazon.rename_column("text", "sentence")

    return amazon


benchmark_datasets = {
    "en": {
        "sent_eval": load_sent_eval,
        "sentiment140": load_sentiment140,
        "mteb": lambda: load_mteb("english"),
        "amazon": lambda: load_amazon("en"),
        "sst2": lambda: load_dataset("stanfordnlp/sst2")["validation"],
        "financial_phrasebank": lambda: load_dataset(
            "takala/financial_phrasebank", "sentences_66agree"
        )["train"],
    },
    "es": {
        "mteb": lambda: load_mteb("spanish"),
        "amazon": lambda: load_amazon("es"),
    },
}


class PySentimientoAnalyzer:
    def __init__(self, lang):
        self.analyzer = create_analyzer("sentiment", lang=lang)

    def __call__(self, dataset):
        id2label = dataset.features["label"].names

        outs = self.analyzer.predict(dataset["sentence"])

        if len(id2label) == 2:
            # Only positive/negative
            return [
                "negative" if x.probas["NEG"] > x.probas["POS"] else "positive"
                for x in outs
            ]
        else:
            translation = {"NEU": "neutral", "POS": "positive", "NEG": "negative"}
            return [translation[x.output] for x in outs]


class StanzaAnalyzer:
    def __init__(self, lang):
        self.nlp = stanza.Pipeline(
            lang=lang, processors="tokenize,sentiment", tokenize_no_ssplit=True
        )

    def __call__(self, dataset):
        id2label = dataset.features["label"].names
        outs = self.nlp(dataset["sentence"])

        def _get_sentiment(x):
            if x.sentiment == 0:
                return "negative"
            elif x.sentiment == 2:
                return "positive"
            elif len(id2label) == 2:
                # Flip a coin
                if random.random() > 0.5:
                    return "positive"
                else:
                    return "negative"
            else:
                return "neutral"

        return [_get_sentiment(x) for x in outs.sentences]


class TweetNLPAnalyzer:
    def __init__(self, lang):
        pass

    def __call__(self, dataset):
        # Load here, this model has a memory leak
        self.model = tweetnlp.load_model("sentiment")
        id2label = dataset.features["label"].names

        logger.info(f"Slow model -- running one by one")
        # TweetNLP runs OOM if we run all at once
        outs = [self.model.predict(ex["sentence"]) for ex in tqdm(dataset)]

        del self.model

        def get_tweetnlp_sentiment(x):
            if x["label"] in {"positive", "negative"}:
                return x["label"]
            elif len(id2label) == 2:
                # Flip a coin
                if random.random() > 0.5:
                    return "positive"
                else:
                    return "negative"
            else:
                return "neutral"

        return [get_tweetnlp_sentiment(x) for x in outs]


class TextBlobAnalyzer:
    def __init__(self, lang):
        pass

    def __call__(self, dataset):
        id2label = dataset.features["label"].names
        outs = [TextBlob(x).sentiment.polarity for x in dataset["sentence"]]

        def get_textblob_sentiment(x):
            if len(id2label) == 2:
                if x > 0:
                    return "positive"
                else:
                    return "negative"
            else:
                if x > 0.1:
                    return "positive"
                elif x < -0.1:
                    return "negative"
                else:
                    return "neutral"

        return [get_textblob_sentiment(x) for x in outs]


class VaderAnalyzer:
    def __init__(self, lang):
        pass

    def __call__(self, dataset):
        id2label = dataset.features["label"].names
        vader = SentimentIntensityAnalyzer()
        outs = [vader.polarity_scores(x) for x in dataset["sentence"]]

        def get_vader_sentiment(x):
            if len(id2label) == 2:
                if x["pos"] > x["neg"]:
                    return "positive"
                else:
                    return "negative"
            else:
                labels = ["neg", "neu", "pos"]

                # get argmax
                max_sent = max(range(len(labels)), key=lambda i: x[labels[i]])

                return id2label[max_sent]

        return [get_vader_sentiment(x) for x in outs]


class FlairAnalyzer:
    def __init__(self, lang):
        if lang != "en":
            raise ValueError("Flair only supports English")
        self.tagger = Classifier.load("sentiment")

    def __call__(self, dataset):
        id2label = dataset.features["label"].names

        if len(id2label) > 2:
            raise ValueError("Flair only supports binary classification")

        sentences = [Sentence(x) for x in dataset["sentence"]]

        self.tagger.predict(sentences)

        def get_label(sent):
            labels = sent.annotation_layers["label"]
            # Get the one with highest value

            annot = max(labels, key=lambda x: x._score)
            return annot.value.lower()

        outs = [get_label(x) for x in sentences]

        return outs


allowed_models = {
    "en": ["vader", "textblob", "stanza", "tweetnlp", "pysentimiento", "flair"],
    "es": ["stanza", "tweetnlp", "pysentimiento"],
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", help="Dataset to evaluate", type=str, default=None)
    parser.add_argument("--lang", type=str, required=True, help="Language")
    parser.add_argument("--output", type=str, help="Output file", required=True)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to evaluate",
    )
    args = parser.parse_args()

    analyzers = {
        "vader": VaderAnalyzer,
        "textblob": TextBlobAnalyzer,
        "stanza": StanzaAnalyzer,
        "tweetnlp": TweetNLPAnalyzer,
        "pysentimiento": PySentimientoAnalyzer,
        "flair": FlairAnalyzer,
    }

    lang = args.lang

    if args.dataset is None:
        eval_datasets = list(benchmark_datasets[lang].keys())
    else:
        eval_datasets = [args.dataset]

    if args.model not in allowed_models[lang]:
        raise ValueError(f"Model {args.model} not available for language {lang}")

    analyzer = analyzers[args.model](args.lang)

    logger.info(f"Benchmarking {args.model} on {eval_datasets}")

    results = []

    for ds_name in tqdm(eval_datasets):
        print(ds_name)
        dataset = benchmark_datasets[lang][ds_name]()
        try:
            preds = analyzer(dataset)

            id2label = dataset.features["label"].names
            label2id = {v: k for k, v in enumerate(id2label)}
            true_labels = dataset["label"]
            pred_labels = [label2id[x] for x in preds]

            ret = classification_report(
                true_labels, pred_labels, target_names=id2label, output_dict=True
            )

            res = {
                "Model": args.model,
                "Dataset": ds_name,
                "Macro F1": ret["macro avg"]["f1-score"],
                "Macro Precision": ret["macro avg"]["precision"],
                "Macro Recall": ret["macro avg"]["recall"],
            }

            results.append(res)
        except ValueError as e:
            logger.error(f"Error on {ds_name}: {e}")

    logger.info(results)

    pd.DataFrame(results).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
