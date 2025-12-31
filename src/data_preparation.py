import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def data_prepare(x_train, x_test):
    tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")

    tokenizer.fit_on_texts(x_train)
    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_test_seq = tokenizer.texts_to_sequences(x_test)
    
    x_train_pad = pad_sequences(x_train_seq, maxlen=200, padding='post')
    x_test_pad = pad_sequences(x_test_seq, maxlen=200, padding='post')
    
    with open("artifacts/tokenizer.pkl", 'wb') as f:
        pickle.dump(tokenizer, f)
    
    return x_train_pad,x_test_pad