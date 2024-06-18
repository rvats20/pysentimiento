# Comparison with other opinion-mining tools

We compare `pysentimiento` with other opinion-mining toolkits, namely:

- [VADER](https://github.com/cjhutto/vaderSentiment)
- [TextBlob](https://textblob.readthedocs.io/)
- [Stanza](https://stanfordnlp.github.io/stanza/sentiment.html)
- [TweetNLP](https://github.com/cardiffnlp/tweetnlp/)

See the [comparison notebook](Results.ipynb)

## Languages and datasets


For Italian, datasets available at the [Huggingface hub](https://huggingface.co/datasets?language=language:it&sort=trending&search=senti) are mostly based on SentiPOLC, the dataset used to train the model. 


## Running experiments

Run

```bash
./run_comparison.sh
```

to run the comparison experiments.