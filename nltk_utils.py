# import numpy as np
# import nltk
# from nltk.stem.porter import PorterStemmer
# stemmer = PorterStemmer()

# def tokenize(sentence):
#     """
#     Membagi kalimat menjadi daftar kata atau token.
#     """
#     return nltk.word_tokenize(sentence)

# def stem(word):
#     """
#     Mengubah kata menjadi bentuk dasarnya.
#     """
#     return stemmer.stem(word.lower())

# def bag_of_words(tokenized_sentence, words):
#     """
#     Mengubah kalimat yang sudah ditokenisasi menjadi representasi numerik.
#     example:
#     sentence = ["hello", "how", "are", "you"]
#     words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
#     bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
#     """
#     # Stemming untuk tiap kata
#     sentence_words = [stem(word) for word in tokenized_sentence]
#     # Inisialisasi bag 
#     bag = np.zeros(len(words), dtype=np.float32)
#     for idx, w in enumerate(words):
#         if w in sentence_words: 
#             bag[idx] = 1

#     return bag

import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Tokenizer
def tokenize(sentence):
    return word_tokenize(sentence)

# Stemmer (Sastrawi)
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stem(word):
    return stemmer.stem(word.lower())

# Encode kalimat ke index (pakai word2idx)
def encode_sentence(sentence, word2idx):
    tokens = tokenize(sentence)
    tokens = [stem(w) for w in tokens]
    encoded = [word2idx[w] for w in tokens if w in word2idx]
    return encoded
