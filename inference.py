import pickle
import keras
import re
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Input, Embedding, Dense
from keras.models import Model
import tensorflow as tf
import numpy as np

tf.config.set_visible_devices([], 'GPU')

with open('input_tokenizer (2).pickle', 'rb') as handle:
    tokenizer_articles = pickle.load(handle)
with open('target_tokenizer (2).pickle', 'rb') as handle:
    tokenizer_summaries = pickle.load(handle)

summaries_vocab = len(tokenizer_summaries.word_index) + 1

model = keras.saving.load_model('newsSummarizer_modelNEWMODEL20.keras')
model.summary()

#Encoder--------------------------------------------------------------------------------------------------------------------

encoder_inputs = model.input[0]

encoder_embedding = model.get_layer("encoder_embedding")
encoder_embedded = encoder_embedding(encoder_inputs)

lstm1 = model.get_layer('lstm')
lstm_output,h1,c1 = lstm1(encoder_embedded)

lstm2 = model.get_layer('lstm_1')
lstm_output2,h2,c2 = lstm2(lstm_output)

lstm3 = model.get_layer('lstm_2')
lstm_output3,h3,c3 = lstm3(lstm_output2)

encoder_states1 = [h3, c3]
encoder_model = keras.Model(encoder_inputs, encoder_states1)

# DECODER
decoder_inputs = model.get_layer('decoder_inputs').input
decoder_state_input_h = keras.Input(shape=(350,), name="decoder_state_input_h")
decoder_state_input_c = keras.Input(shape=(350,), name="decoder_state_input_c")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_embedding = model.get_layer('decoder_embedding')
embeddings = decoder_embedding(decoder_inputs)

decoder_lstm = model.get_layer('lstm_3')
decoder_outputs1, state_h_dec, state_c_dec = decoder_lstm(embeddings, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]

decoder_dense = model.get_layer('dense')
decoder_outputs = decoder_dense(decoder_outputs1)

decoder_model = keras.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

def stripping_data(row):
    pattern = r'[^a-zA-Z0-9\s\.,!?:;\'\"\-\(\)\[\]]'

    row = re.sub(pattern, '', str(row)).lower()
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
    return row

def preprocess_new_article(article):
    article = stripping_data(article)

    article_sequence = tokenizer_articles.texts_to_sequences([article])

    padded_article = pad_sequences(article_sequence, maxlen=3000, padding='post')
    return padded_article


def inference(input_text, max_decoder_seq_length= 100):

    input_seq = np.array(input_text)

    encoder_h, encoder_c = encoder_model.predict(input_seq)
    states_value = [encoder_h,encoder_c]

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer_summaries.word_index["<start>"]

    decoded_sentence = ''

    for _ in range(max_decoder_seq_length):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index in tokenizer_articles.index_word:
            word = tokenizer_articles.index_word[sampled_token_index]

            decoded_sentence += word + ' '

            target_seq = np.zeros((1, 1))
            target_seq[0, 0]= sampled_token_index

        states_value = [h, c]

    return decoded_sentence

# output = preprocess_new_article(new_article)
# print(inference(output))
