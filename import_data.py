# %%
import numpy as np
import pandas as pd
# import torch 
import os
path = 'D://cs6901_assignment_3//aksharantar_sampled//'
dirs = os.listdir(path)

# %%
train_df = pd.DataFrame()
for dir in dirs:
    train_df = pd.concat([train_df, pd.read_csv(path+dir+'//'+dir+'_train.csv', header=None)], axis=0)
print(train_df.head(),len(train_df))

# %%
val_df = pd.DataFrame()
for dir in dirs:
    val_df = pd.concat([val_df, pd.read_csv(path+dir+'//'+dir+'_valid.csv', header=None)], axis=0)
print(val_df.head(),len(val_df))

# %%
test_df = pd.DataFrame()
for dir in dirs:
    test_df = pd.concat([test_df, pd.read_csv(path+dir+'//'+dir+'_test.csv', header=None)], axis=0)
print(test_df.head(),len(test_df))

# %%
# Step 1: Tokenize the input and target sentences
input_sentences = list(train_df[0].values)
target_sentences = list(train_df[1].values)
input_tokens = [sentence.split() for sentence in input_sentences]
target_tokens = [sentence.split() for sentence in target_sentences]

# Step 2: Create vocabulary
input_vocab = set([word for sentence in input_tokens for word in sentence])
target_vocab = set([word for sentence in target_tokens for word in sentence])
# %%
# Step 3: Convert words to indices
input_word2index = {word: i+2 for i, word in enumerate(input_vocab)}
input_word2index['<PAD>'] = 0
input_word2index['<SOS>'] = 1
# %%
target_word2index = {word: i+2 for i, word in enumerate(target_vocab)}
target_word2index['<PAD>'] = 0
target_word2index['<SOS>'] = 1
# %%
# Convert tokens to indices
input_indices = [[input_word2index[word] for word in sentence] for sentence in input_tokens]
target_indices = [[target_word2index[word] for word in sentence] for sentence in target_tokens]
# %%
# Step 4: Pad sequences
max_input_len = max([len(sentence) for sentence in input_indices])
max_target_len = max([len(sentence) for sentence in target_indices])

input_padded = [sentence + [input_word2index['<PAD>']] * (max_input_len - len(sentence)) for sentence in input_indices]
target_padded = [sentence + [target_word2index['<PAD>']] * (max_target_len - len(sentence)) for sentence in target_indices]
# %%
