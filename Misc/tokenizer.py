# -*- coding: utf-8 -*-
"""
Task Tokenizer Demo

Created on Mon November 24 12:05:14 2025

@author: agha
"""

import nltk
nltk.download("punkt")

from nltk.tokenize import word_tokenize

string = "an apologetic fox snatched the grape"
tokens = word_tokenize(string)
print('NLTK tokenizer: ', tokens)

# Tiktoken
import tiktoken
# enc = tiktoken.encoding_for_model("gpt-4o")
enc = tiktoken.get_encoding("o200k_base") # cl100k_base is used for GPT-4, GPT-4o, GPT-3.5-Turbo
print('Number of vocabularies: ', enc.n_vocab)

encoded_string = enc.encode(string)
decoded_string = enc.decode(encoded_string)
print(encoded_string)
print(decoded_string)

print('Tiktoken tokenizer: ')
for token_id in encoded_string:
    token_bytes = enc.decode_bytes([token_id])
    try:
        token_text = token_bytes.decode("utf-8")
    except UnicodeDecodeError:
        token_text = str(token_bytes)
    print(token_text)

