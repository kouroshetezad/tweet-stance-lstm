# Tweet Stance Classification

Stance detection on the SemEval-style stance dataset. Compares classical text-classification baselines against a multi-input LSTM that also conditions on the tweet **target**.

## Pipeline

### 1. Loading & EDA
- Reads `raw_*.csv` splits (train / val / test), tags each with its `type`, and concatenates.
- Renames `Tweet` → `text` and `Stance` → `label`.
- Prints class balance, target distribution, and target × label crosstabs.
- Removes duplicate and null tweets.

### 2. Preprocessing
- Lowercasing, URL / mention removal, non-alphanumeric stripping.
- NLTK tokenization and English stopword removal.

### 3. Baselines
All with `LogisticRegression` (max_iter = 500, random_state = 42), 5000 features, n-grams (1, 2):

| Feature | Notes |
|---|---|
| BoW | `CountVectorizer` |
| TF-IDF | `TfidfVectorizer` |
| Word2Vec | 300-dim gensim vectors, averaged per tweet |

### 4. Multi-input LSTM
- Tokenized + padded tweets (max length 50, vocab 10 000).
- Embedding layer initialized from the trained Word2Vec weights.
- Two stacked LSTM layers (64 → 32, dropout 0.3).
- The categorical **target** is one-hot encoded and concatenated with the LSTM output.
- Dense head (64 → 32) with a sigmoid output for binary stance.
- Adam (lr = 0.001), binary cross-entropy, class weights for imbalance, early stopping.
- Reports test accuracy, a classification report, a confusion matrix, and correct/incorrect sample predictions.

## Requirements

```
pandas numpy nltk gensim scikit-learn tensorflow matplotlib seaborn
```

Data files (`raw_train.csv`, `raw_val.csv`, `raw_test.csv`) are expected in a `content/` directory — adjust the `glob` path at the top of the script if needed.

## Run

```bash
python tweet-stance-classification.py
```
