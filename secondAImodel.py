import keras
import pandas as pd
import re
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.src.callbacks import EarlyStopping
from keras.src.layers import MaxPooling1D
from keras.src.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.utils.class_weight import compute_class_weight
import nltk
from nltk.stem import PorterStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Dense, LSTM, Flatten, Conv1D,MaxPooling1D, GlobalMaxPooling1D
from keras.models import Model
import numpy as np


def importingData():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    #
    data = pd.read_csv('CategoryData.csv')
    print(data.head(5))
    return data


def stripping_data(row):
    row = re.sub("(\\t)", ' ', str(row)).lower()
    row = re.sub("(\\r)", ' ', str(row)).lower()
    row = re.sub("(\\n)", ' ', str(row)).lower()

    row = re.sub("(__+)", ' ', str(row)).lower()
    row = re.sub("(--+)", ' ', str(row)).lower()
    row = re.sub("(~~+)", ' ', str(row)).lower()
    row = re.sub("(\+\++)", ' ', str(row)).lower()
    row = re.sub("(\.\.+)", ' ', str(row)).lower()
    row = re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", ' ', str(row)).lower()
    row = re.sub("(\s+)", ' ', str(row)).lower()
    # Might have to remove single characters with spaces before and after it cause words such as there's become there s
    return row


def stopWords(data):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(data)
    new_sentence = ""
    for word in word_tokens:
        if word.lower() not in stop_words:
            new_sentence = new_sentence + " " + word

    return new_sentence

def stemLemStopTokenizer(data):
    num_for_dense = len(pd.unique(data["categories"]))
    print(num_for_dense)

    print(data.head(5))

    stop_words = set(stopwords.words('english'))

    for i, sentences in enumerate(data["articles"].tolist()):
        word_tokens = word_tokenize(sentences)

        new_sentence = ""
        for w in word_tokens:
            if w.lower() not in stop_words:
                new_sentence = new_sentence + " " + w

        data.iloc[i,2] = new_sentence



    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    data["articles"] = data["articles"].apply(lambda x: stemmer.stem(x))
    data["articles"] = data["articles"].apply(lambda x: stripping_data(x))
    data["articles"] = data["articles"].apply(lambda x: lemmatizer.lemmatize(x))

    print(data.head(5))

    max_len_of_articles = max([len(text.split()) for text in data['articles']])
    tokenizer_headlines_for_categories = Tokenizer()

    tokenizer_headlines_for_categories.fit_on_texts(data["articles"])
    vocab_size = len(tokenizer_headlines_for_categories.word_index) + 1
    sequences = tokenizer_headlines_for_categories.texts_to_sequences(data["articles"])
    sequences = pad_sequences(sequences, maxlen=2000, padding='post')
    print(sequences)

    with open("tokenizer_headlines.pickle", 'wb') as handle:
        pickle.dump(tokenizer_headlines_for_categories, handle, protocol=pickle.HIGHEST_PROTOCOL)

    le = LabelEncoder()
    y_labled = le.fit_transform(data["categories"])
    print(y_labled)
    with open("lableEnconder.pickle", 'wb') as handle:
        pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return num_for_dense, sequences, y_labled, vocab_size, le, max_len_of_articles


def modelTraining(sequences, y_labled, vocab_size, num_for_dense,max_len_of_articles):
    X, x_test, Y, y_test = train_test_split(sequences, y_labled, test_size=0.10, random_state=3)
    print(X)
    X_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.20, random_state=2)


    inputs = Input(shape=(None,), dtype="int64")
    x = Embedding(vocab_size, 50)(inputs)
    x = Conv1D(filters=50, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    output = Dense(num_for_dense, activation='softmax')(x)
    model = Model(inputs, output)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(X_train, y_train,
              batch_size=8,
              epochs=20,
              validation_data=(x_val, y_val))

    model.save('newsCategoriesNEW.keras')


def gettingCategories(headline):
    print(headline)

    with open('tokenizer_headlines.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    with open('lableEnconder.pickle', 'rb') as handle:
        lableEncoder = pickle.load(handle)

    lemmatizer2 = WordNetLemmatizer()
    stemmer2 = PorterStemmer()

    stripped_Data = stripping_data(headline)
    stop_worded = stopWords(stripped_Data)

    word_tokens = word_tokenize(stop_worded)

    processed_words = []
    for word in word_tokens:
        stemmed_word = stemmer2.stem(word)
        lemmed_word = lemmatizer2.lemmatize(stemmed_word)
        processed_words.append(lemmed_word)

    processed_headline = ' '.join(processed_words)

    sequenced = tokenizer.texts_to_sequences([processed_headline])
    padded = pad_sequences(sequenced, maxlen=2000, padding='post')

    model = keras.saving.load_model("newsCategoriesNEW.keras")
    categoryClass = model.predict(padded)

    print(categoryClass.shape)
    categoryIndex = np.argmax(categoryClass, axis=-1)
    print(categoryIndex)

    labelValue = lableEncoder.inverse_transform(categoryIndex)
    print(labelValue[0])

    return labelValue[0]


# data = importingData()
# num_for_dense, sequences, y_labled, vocab_size, le,max_len_of_articles = stemLemStopTokenizer(data)
# modelTraining(sequences,y_labled,vocab_size,num_for_dense,max_len_of_articles)
