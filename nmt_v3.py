# -*- coding: utf-8 -*-

import os
import datetime
from google.colab import drive
drive.mount('/content/gdrive')

os.chdir('/content/gdrive/My Drive/Deep Learning CS6910/rnn_test')

# !pip install --upgrade wandb -qq

import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import wandb
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output

device_gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path = './aksharantar_sampled/'
dirs = os.listdir(path)
# dirs

train_df = pd.read_csv(path+'hin'+'/'+'hin_train.csv', header=None)
print(train_df.head())
print(len(train_df))
train_data = train_df.values.T
train_data.shape

valid_df = pd.read_csv(path+'hin'+'/'+'hin_valid.csv', header=None)
print(valid_df.head())
print(len(valid_df))
valid_data = valid_df.values

test_df = pd.read_csv(path+'hin'+'/'+'hin_valid.csv', header=None)
print(test_df.head())
print(len(test_df))
test_data = test_df.values.T

# storing all the alphabets of English and the pad char to a dictionary to create OHE representation later.
eng_alphabets = 'abcdefghijklmnopqrstuvwxyz'
pad_char = '-PAD-'

eng_alpha2index = {pad_char: 0}
for index, alpha in enumerate(eng_alphabets):
    eng_alpha2index[alpha] = index+1

print(eng_alpha2index)

tensor_dict = {key: torch.tensor(value).to(device_gpu) for key, value in eng_alpha2index.items()}

hindi_alphabets = [chr(alpha) for alpha in range(2304, 2432)]
hindi_alphabet_size = len(hindi_alphabets)

hindi_alpha2index = {pad_char: 0}
for index, alpha in enumerate(hindi_alphabets):
    hindi_alpha2index[alpha] = index+1

print(hindi_alpha2index)

import re
non_eng_letters_regex = re.compile('[^a-zA-Z ]')

# Remove all English non-letters
def cleanEnglishVocab(line):
    line = line.replace('-', ' ').replace(',', ' ').upper()
    line = non_eng_letters_regex.sub('', line)
    return line.split()

# Remove all Hindi non-letters
def cleanHindiVocab(line):
    line = line.replace('-', ' ').replace(',', ' ')
    cleaned_line = ''
    for char in line:
        if char in hindi_alpha2index or char == ' ':
            cleaned_line += char
    return cleaned_line.split()

class TransliterationDataLoader(Dataset):
    def __init__(self, eng_words, hindi_words):
        self.eng_words = eng_words
        self.hindi_words = hindi_words
        self.shuffle_indices = list(range(len(self.eng_words)))
        random.shuffle(self.shuffle_indices)
        self.shuffle_start_index = 0
        
    def __len__(self):
        return len(self.eng_words)
    
    def __getitem__(self, idx):
        return self.eng_words[idx], self.hindi_words[idx]
    
    def get_random_sample(self):
        return self.__getitem__(np.random.randint(len(self.eng_words)))
    
    def get_batch_from_array(self, batch_size, array):
        end = self.shuffle_start_index + batch_size
        batch = []
        if end >= len(self.eng_words):
            batch = [array[i] for i in self.shuffle_indices[0:end % len(self.eng_words)]]
            end = len(self.eng_words)
        return batch + [array[i] for i in self.shuffle_indices[self.shuffle_start_index : end]]
    
    def get_batch(self, batch_size, postprocess=True):
        eng_batch = self.get_batch_from_array(batch_size, self.eng_words)
        hindi_batch = self.get_batch_from_array(batch_size, self.hindi_words)
        self.shuffle_start_index += batch_size + 1
        
        if self.shuffle_start_index >= len(self.eng_words):
            random.shuffle(self.shuffle_indices)
            self.shuffle_start_index = 0
            
        return eng_batch, hindi_batch

train_data = TransliterationDataLoader(train_data[0],train_data[1])

for i in range(10):
  eng, hindi = train_data.get_random_sample()
  print(eng + ' - ' + hindi)

def word_rep(word, letter2index, device = 'cpu'):
    rep = torch.zeros(len(word)+1, 1, len(letter2index)).to(device)
    for letter_index, letter in enumerate(word):
        pos = letter2index[letter]
        rep[letter_index][0][pos] = 1
    pad_pos = letter2index[pad_char]
    rep[letter_index+1][0][pad_pos] = 1
    return rep

def gt_rep(word, letter2index, device = 'cpu'):
    gt_rep = torch.zeros([len(word)+1, 1], dtype=torch.long).to(device)
    for letter_index, letter in enumerate(word):
        pos = letter2index[letter]
        gt_rep[letter_index][0] = pos
    gt_rep[letter_index+1][0] = letter2index[pad_char]
    return gt_rep

eng, hindi = train_data.get_random_sample()
eng_rep = word_rep(eng, eng_alpha2index)
# print(eng, eng_rep)

hindi_gt = gt_rep(hindi, hindi_alpha2index)
print(hindi, hindi_gt.shape[0])

MAX_OUTPUT_CHARS = 30

class Transliteration_EncoderDecoder_BeamSearch(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, type_ = 'rnn', bidirectional=False, beam_width=5,embedding_size=128, num_layers=1, verbose=False):
    super(Transliteration_EncoderDecoder_BeamSearch, self).__init__()
    
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.beam_width = beam_width
    self.type_ = type_


    self.embedding = nn.Embedding(input_size, embedding_size)
    
    if type_ == 'gru':
      self.encoder_rnn_cell = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,bidirectional=bidirectional)
      self.decoder_rnn_cell = nn.GRU(input_size=output_size, hidden_size=hidden_size,num_layers=num_layers,bidirectional=bidirectional)
    
    elif type_ == 'rnn':
      self.encoder_rnn_cell = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,bidirectional=bidirectional)
      self.decoder_rnn_cell = nn.RNN(input_size=output_size, hidden_size=hidden_size,num_layers=num_layers,bidirectional=bidirectional)
    
    elif type_ == 'lstm':
      self.encoder_rnn_cell = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,bidirectional=bidirectional)
      self.decoder_rnn_cell = nn.LSTM(input_size=output_size, hidden_size=hidden_size,num_layers=num_layers,bidirectional=bidirectional)
    
    self.h2o = nn.Linear(hidden_size, output_size)
    self.softmax = nn.LogSoftmax(dim=2)
    
    self.verbose = verbose
  
  def forward(self, input, max_output_chars=MAX_OUTPUT_CHARS, device='cpu', ground_truth=None):
    # encoder
    # Convert input tensor to LongTensor
    input = input.long()

    # Embedding
    embedded_input = self.embedding(input)
    embedded_input = embedded_input.view((embedded_input.shape[0]*embedded_input.shape[2],1,embedded_input.shape[3]))
    if self.type_ == 'lstm':
      out, (hidden,_) = self.encoder_rnn_cell(embedded_input)
    else:  
      out, hidden = self.encoder_rnn_cell(embedded_input)
    
    if self.verbose:
      print('Encoder input', input.shape)
      print('Encoder output', out.shape)
      print('Encoder hidden', hidden.shape)
    
    # decoder
    decoder_state = hidden
    decoder_input = torch.zeros(1, 1, self.output_size).to(device)
    decoder_state_ = torch.zeros_like(decoder_state).to(device)
    outputs = []
    
    if self.verbose:
      print('Decoder state', decoder_state.shape)
      print('Decoder input', decoder_input.shape)
    
    for i in range(max_output_chars):
      if self.type_ == 'lstm':
        out, (decoder_state,_) = self.decoder_rnn_cell(decoder_input, (decoder_state,decoder_state_))
      else:
        out,decoder_state = self.decoder_rnn_cell(decoder_input,decoder_state)
      
      if self.verbose:
        print('Decoder intermediate output', out.shape)
      
      out = self.h2o(decoder_state)
      out = self.softmax(out)
      outputs.append(out.view(1, -1))
      
      if self.verbose:
          print('Decoder output', out.shape)
          self.verbose = False
      
      if ground_truth is not None:
          max_idx = ground_truth[i].reshape(1, 1, 1)
      else:
          topk_probs, topk_indices = out.topk(self.beam_width, dim=2)
          topk_probs = topk_probs.view(1, -1)
          topk_indices = topk_indices.view(1, -1)
          # Exclude invalid probabilities
          topk_probs[torch.isnan(topk_probs)] = 0
          topk_probs[torch.isinf(topk_probs)] = 0
          # Normalize probabilities
          topk_probs /= topk_probs.sum()
          # Perform multinomial sampling
          selected_indices = torch.multinomial(topk_probs, 1)
          max_idx = topk_indices[0][selected_indices[0]].reshape(1, 1, 1)
      
      one_hot = torch.FloatTensor(out.shape).to(device)
      one_hot.zero_()
      one_hot.scatter_(2, max_idx, 1)
      
      decoder_input = one_hot.detach()
    
    return outputs

def infer(net, eng_word,shape,device ='cpu'):
  # net.eval()
  net.to(device)
  input_ = word_rep(eng_word,tensor_dict,device) # convert the name into one hot encoding.
  outputs = net(input_,shape,device) # initilise the hidden layer.
  
  return outputs

net = Transliteration_EncoderDecoder_BeamSearch(len(eng_alpha2index), 256, len(hindi_alpha2index), verbose=True,bidirectional=True, num_layers=2)

out = infer(net, 'india', 30, device_gpu)
# type(out)

def train_batch(net, opt, criterion, batch_size, device = 'cpu', teacher_force = False):
    
  net.train().to(device)
  opt.zero_grad()
  eng_batch, hindi_batch = train_data.get_batch(batch_size)
  
  total_loss = 0
  accuracy = 0
  for i in range(batch_size):
      
    input = word_rep(eng_batch[i], eng_alpha2index, device)
    gt = gt_rep(hindi_batch[i], hindi_alpha2index, device)
    outputs = net(input, gt.shape[0], device, ground_truth = gt if teacher_force else None)
    
    correct = 0
    for index, output in enumerate(outputs):
      loss = criterion(output, gt[index]) / batch_size
      loss.backward(retain_graph = True)
      total_loss += loss

      val, indices = output.topk(1)
      hindi_pos = indices.tolist()[0]
      if hindi_pos[0] == gt[index][0]:
        correct += 1
    accuracy += correct/gt.shape[0]
  accuracy /= batch_size
  opt.step()


  return total_loss.cpu().detach().numpy()/batch_size,accuracy

def train_setup(net, lr = 0.01, n_batches = 100, batch_size = 10, momentum = 0.9, display_freq=5, device = 'cpu',name='model'):

  log = {}
    
  net = net.to(device)
  criterion = nn.NLLLoss(ignore_index = -1)
  opt = optim.Adam(net.parameters(), lr=lr)
  teacher_force_upto = n_batches//3
  
  # loss_arr = np.zeros(n_batches + 1)
  
  for i in range(n_batches):
    loss,accuracy = train_batch(net, opt, criterion, batch_size, device = device, teacher_force = i<teacher_force_upto )

    log['loss'] = loss
    log['acc'] = accuracy
    
    # if i%display_freq == display_freq-1:
    #     clear_output(wait=True)
        
        # print('Iteration', i, 'Loss', loss,'accuracy:',accuracy)
        # plt.figure()
        # plt.plot(loss, '-*')
        # plt.xlabel('Iteration')
        # plt.ylabel('Loss')
        # plt.show()
        # print('\n\n')

    val_acc = calc_accuracy(net,valid_data) 
    log['val_acc'] = val_acc
    wandb.log(log)
    # print('val_acc',val_acc)    
  torch.save(net, name+'.pt')
  # return loss

net = Transliteration_EncoderDecoder_BeamSearch(len(eng_alpha2index), 256, len(hindi_alpha2index))

train_setup(net, lr=0.001, n_batches=20, batch_size = 1, display_freq=5, device = device_gpu)

def test(net, word, device = 'cpu'):
  net = net.eval().to(device)
  outputs = infer(net, word, 30, device)
  hindi_output = ''
  for out in outputs:
      val, indices = out.topk(1)
      index = indices.tolist()[0][0]
      if index == 0:
          break
      hindi_char = hindi_alphabets[index+1]
      hindi_output += hindi_char
  print(word + ' - ' + hindi_output)
  return hindi_output

net = Transliteration_EncoderDecoder_BeamSearch(len(eng_alpha2index), 16, len(hindi_alpha2index),type_='lstm',beam_width=5,bidirectional=True,embedding_size=256,num_layers=1)

net =torch.load('treasured-sweep-2.pt')

test(net,'test')

def calc_accuracy(net,Data, device = 'cpu'):
  net = net.eval()#.to('cpu')
  predictions = []
  accuracy = 0
  for i in range(len(Data)):
    data = Data[i]
    eng, hindi = data[0],data[1]
    gt = gt_rep(hindi, hindi_alpha2index, device_gpu)
    outputs = infer(net, eng, gt.shape[0], device_gpu)
    correct = 0
    for index, out in enumerate(outputs):
      val, indices = out.topk(1)
      hindi_pos = indices.tolist()[0]
      if hindi_pos[0] == gt[index][0]:
        correct += 1      
    accuracy += correct/gt.shape[0]
  accuracy /= len(Data)
  # print(accuracy)
  return accuracy

calc_accuracy(net,valid_data)

sweep_config = {
    'method': 'bayes', 
    'metric': {
      'name': 'val_acc',
      'goal': 'maximize'   
    },
    'parameters':{
        'embedding_size':{
            'values':[16,32,64,128,256]
        },
        'n_layers':{
            'values':[1,2,3]
        },
        'hidden_size':{
            'values':[16, 32, 64, 256],
        },
        'cell_type':{
            'values':['rnn','lstm','gru']
        },
        'bidirectional':{
            'values':[True,False]
        },
        'dropout':{
            'values':[0.2,0.3]
        },
        'beam_width':{
            'values':[3,4,5]
        }
    }
    }

sweep_id = wandb.sweep(sweep_config, entity='viswa_ee', project="CS6910_NLG")

def train():
  config_defaults={
      'embedding_size':16,
      'n_layers':1,
      'hidden_size':16,
      'cell_type':'gru',
      'bidirectional':False,
      'dropout':0.2,
      'beam_width':2
  }
  wandb.init(config=config_defaults)
  config = wandb.config
  net = Transliteration_EncoderDecoder_BeamSearch(input_size=len(eng_alpha2index), hidden_size=256, output_size=len(hindi_alpha2index),type_=config.cell_type,bidirectional=config.bidirectional,
                                                  beam_width=config.beam_width,embedding_size=config.embedding_size,num_layers=config.n_layers)
  
  train_setup(net, lr=0.001, n_batches=50, batch_size = 64, display_freq=5, device = device_gpu,name=wandb.run.name)

wandb.agent(sweep_id,function=train,count=20)

