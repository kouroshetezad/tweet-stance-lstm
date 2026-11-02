import pandas as pd
import glob
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
nltk.download('stopwords')
from tensorflow.keras.regularizers import l2
nltk.download('punkt_tab')
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Flatten, Bidirectional, Dropout
from sklearn.utils import class_weight
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping


#part 1
files = glob.glob("/content/raw_*.csv")

all_data = []

for f in files:
    parts = f.split("_")
    data_type = parts[1].split(".")[0]

    df = pd.read_csv(f)
    df["type"] = data_type
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

data.rename(columns={'Tweet': 'text', 'Stance': 'label'}, inplace=True)

print("\n number of against and favor: \n", data['label'].value_counts())
print("\n percentage of against and favor: \n", data['label'].value_counts(normalize=True) * 100)

print("\n distribution of targets: \n", data['Target'].value_counts())
print("\n percentage distribution of targets: \n", data['Target'].value_counts(normalize=True) * 100)

print("\n(Target vs. Label):\n", pd.crosstab(data['Target'], data['label']))
print("\n percentage per target \n", pd.crosstab(data['Target'], data['label'], normalize='index') * 100)
print("="*60)

initial_rows = data.shape[0]
print(f"\n before preproccessing: {initial_rows}")

data.drop_duplicates(subset=['text'], inplace=True)
print(f"after removing duplicates {data.shape[0]}")

data.dropna(subset=['text'], inplace=True)
print(f"after removing nulls: {data.shape[0]}")


data['raw_tokens'] = data['text'].apply(lambda x: nltk.word_tokenize(x.lower()))

train_data = data[data["type"] == "train"].copy()
val_data  = data[data["type"] == "val"].copy()
test_data = data[data["type"] == "test"].copy()

stop_words = set(stopwords.words('english'))

#removing stopwords
def tokenize_and_clean(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    words = nltk.word_tokenize(text)
    filtered_words = [w for w in words if w not in stop_words and len(w) > 1]

    return filtered_words


train_data['clean_tokens'] = train_data['text'].apply(tokenize_and_clean)
val_data['clean_tokens']  = val_data['text'].apply(tokenize_and_clean)
test_data['clean_tokens'] = test_data['text'].apply(tokenize_and_clean)

#return to string form for BoW and TF-IDF
X_train_clean = train_data['clean_tokens'].apply(lambda x: " ".join(x))
X_val_clean  = val_data['clean_tokens'].apply(lambda x: " ".join(x))
X_test_clean = test_data['clean_tokens'].apply(lambda x: " ".join(x))

#define target variables
y_train = train_data["label"]
y_val  = val_data["label"]
y_test = test_data["label"]

RESULTS = {}
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)

print("\n--- Feature Extraction and Baseline Evaluation with BoW and TF-IDF ---")
#BoW
bow_vectorizer = CountVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
X_train_bow = bow_vectorizer.fit_transform(X_train_clean)
X_test_bow = bow_vectorizer.transform(X_test_clean)

bow_model = LogisticRegression(max_iter=500, random_state=42)
bow_model.fit(X_train_bow, y_train)
y_pred_bow = bow_model.predict(X_test_bow)

print("\n1.BoW results:")
print(classification_report(y_test, y_pred_bow))
RESULTS["BoW"] = accuracy_score(y_test, y_pred_bow)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_clean)
X_test_tfidf = tfidf_vectorizer.transform(X_test_clean)

tfidf_model = LogisticRegression(max_iter=500, random_state=42)
tfidf_model.fit(X_train_tfidf, y_train)
y_pred_tfidf = tfidf_model.predict(X_test_tfidf)

print("\n2.TF-IDF results:")
print(classification_report(y_test, y_pred_tfidf))
RESULTS["TF-IDF"] = accuracy_score(y_test, y_pred_tfidf)



EMBEDDING_DIM = 300
MIN_COUNT = 5

all_clean_tokens = pd.concat([train_data['clean_tokens'], val_data['clean_tokens'], test_data['clean_tokens']]).tolist()

w2v_model = Word2Vec(
    sentences=all_clean_tokens,
    vector_size=EMBEDDING_DIM,
    window=5,
    min_count=MIN_COUNT,
    workers=4
)


def average_word_vectors(tokens, model, num_features):
    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0
    index2word_set = set(model.wv.index_to_key)

    for word in tokens:
        if word in index2word_set:
            nwords = nwords + 1
            feature_vector = np.add(feature_vector, model.wv[word])

    if nwords:
        feature_vector = np.divide(feature_vector, nwords)

    return feature_vector

def get_avg_feature_vectors(tweets, model, num_features):
    return np.array([
        average_word_vectors(tokens, model, num_features)
        for tokens in tweets
    ])

X_train_w2v = get_avg_feature_vectors(train_data["clean_tokens"], w2v_model, EMBEDDING_DIM)
X_test_w2v  = get_avg_feature_vectors(test_data["clean_tokens"], w2v_model, EMBEDDING_DIM)

w2v_lr_model = LogisticRegression(max_iter=500, random_state=42)
w2v_lr_model.fit(X_train_w2v, y_train)
y_pred_w2v = w2v_lr_model.predict(X_test_w2v)

print("\n3.Word2Vec results: ")
print(classification_report(y_test, y_pred_w2v))
RESULTS["Word2Vec Avg"] = accuracy_score(y_test, y_pred_w2v)

print("\n--- Baseline results---")
print(pd.Series(RESULTS))


MAX_SEQ_LENGTH = 50
MAX_WORDS = 10000
NUM_CLASSES = len(y_train.unique())


all_train_text = train_data['clean_tokens'].apply(lambda x: " ".join(x)).tolist()
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(all_train_text)


X_train_seq = tokenizer.texts_to_sequences(train_data['clean_tokens'].apply(lambda x: " ".join(x)))
X_val_seq  = tokenizer.texts_to_sequences(val_data['clean_tokens'].apply(lambda x: " ".join(x)))
X_test_seq = tokenizer.texts_to_sequences(test_data['clean_tokens'].apply(lambda x: " ".join(x)))

X_train_padded = pad_sequences(X_train_seq, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')
X_val_padded  = pad_sequences(X_val_seq, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded  = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

target_encoder = LabelEncoder()
target_encoder.fit(train_data['Target'])
X_train_target = to_categorical(target_encoder.transform(train_data['Target']))
X_val_target = to_categorical(target_encoder.transform(val_data['Target']))
X_test_target = to_categorical(target_encoder.transform(test_data['Target']))

TARGET_DIM = X_train_target.shape[1]

embedding_matrix = np.zeros((MAX_WORDS, EMBEDDING_DIM))
word_index = tokenizer.word_index

for word, i in word_index.items():
    if i < MAX_WORDS:
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]

print(f"Word2Vec embedding matrix created. Shape: {embedding_matrix.shape}")

print("\n--- Multi-input LSTM model structure---")

input_tweet = Input(shape=(MAX_SEQ_LENGTH,), name='tweet_input')

tweet_embedding = Embedding(input_dim=MAX_WORDS,
                            output_dim=EMBEDDING_DIM,
                            input_length=MAX_SEQ_LENGTH,
                            trainable=True,
                            name='tweet_embedding')(input_tweet)

lstm_1 = LSTM(units=64,
              dropout=0.3,
              return_sequences=True
              )(tweet_embedding)
lstm_out = LSTM(units=32,
                dropout=0.3,
                )(lstm_1)

input_target = Input(shape=(TARGET_DIM,), name='target_input')


merged = Concatenate(name='merge_layer')([lstm_out, input_target])

#fuly connected layers
dense_1 = Dense(units=64, activation='relu', name='dense_1')(merged)
dense_2 = Dense(units=32, activation='relu', name='dense_2')(dense_1)

output_layer = Dense(1, activation='sigmoid', name='output_layer')(dense_2)

#final model definition
model = Model(inputs=[input_tweet, input_target], outputs=output_layer)


custom_adam = Adam(learning_rate=0.001)

model.compile(
    optimizer=custom_adam,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()




BATCH_SIZE = 128
EPOCHS = 26


early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=30,
    restore_best_weights=True
)

class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train_encoded),
    y=y_train_encoded
)
class_weights_dict = dict(enumerate(class_weights))
print("\n--- LSTM training---")
history = model.fit(
    {'tweet_input': X_train_padded, 'target_input': X_train_target},
    y_train_encoded,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=({'tweet_input': X_val_padded, 'target_input': X_val_target}, y_val_encoded),
    callbacks=[early_stop],
    class_weight=class_weights_dict,
    verbose=1
)


print("\n--- Final evaluation on the test suite(Test Set) ---")


loss, accuracy = model.evaluate(
    {'tweet_input': X_test_padded, 'target_input': X_test_target},
    y_test_encoded,
    verbose=0
)
print(f"Test Accuracy (LSTM): {accuracy:.4f}")

#confession matrix
y_pred_probs = model.predict({'tweet_input': X_test_padded, 'target_input': X_test_target})
y_pred_classes = y_pred_probs.round().flatten().astype(int)
y_true_classes = y_test_encoded

cm = confusion_matrix(y_true_classes, y_pred_classes)
class_labels = label_encoder.classes_

print("\n3. (Classification Report):")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_labels))

print("\n4. (Confusion Matrix):")

#show confession matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix - LSTM Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

print("\n5. Correct and incorrect predictions:")

correct_indices = np.where(y_pred_classes == y_true_classes)[0]
if len(correct_indices) > 5:
    sample_correct = np.random.choice(correct_indices, 5, replace=False)

    print("\n--- 5 correct predictions ---")
    for i in sample_correct:
        print(f"Tweet: {test_data.iloc[i]['text']}")
        print(f"Target: {test_data.iloc[i]['Target']}")
        print(f"True Label: {label_encoder.inverse_transform([y_true_classes[i]])[0]}, Predicted: {label_encoder.inverse_transform([y_pred_classes[i]])[0]}")
        print("-" * 20)

incorrect_indices = np.where(y_pred_classes != y_true_classes)[0]
if len(incorrect_indices) > 5:
    sample_incorrect = np.random.choice(incorrect_indices, 5, replace=False)

    print("\n--- 5 incorrect predictions ---")
    for i in sample_incorrect:
        print(f"Tweet: {test_data.iloc[i]['text']}")
        print(f"Target: {test_data.iloc[i]['Target']}")
        print(f"True Label: {label_encoder.inverse_transform([y_true_classes[i]])[0]}, Predicted: {label_encoder.inverse_transform([y_pred_classes[i]])[0]}")
        print("-" * 20)



#val acc , train acc plot 

def plot_learning_curve(history):
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(train_acc) + 1)
    
    plt.figure(figsize=(10, 6))
  
    plt.plot(epochs, train_acc, 'bo-', label='(Training)')

    plt.plot(epochs, val_acc, 'ro-', label='(Validation)')
    
    plt.xlabel('epoch')
    plt.ylabel('acurracy')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_learning_curve(history)