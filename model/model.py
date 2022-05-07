import nltk
from nltk.tokenize import word_tokenize

import sys, re, os
import string, unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils import data

import random
from datetime import datetime


# some configuration
# EMBEDDING_DIM = 100
# LATENT_DIM = 256
# T encoder = 4
class Encoder(nn.Module):
    def __init__(self, embedding_layer, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size  # LATENT_DIM
        self.num_layers = num_layers  # 1 or 2

        self.embedding = embedding_layer  # vocab size x EMBEDDING_DIM
        self.rnn = nn.LSTM(
            embedding_size, hidden_size, num_layers, bidirectional=True
        )  # , dropout=p) # -> T x N x 2*hidden

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        # x shape: (N, T encoder) where N is batch size

        embedding = self.dropout(self.embedding(x))  # (T encoder, N, EMBEDDING_DIM)
        # embedding shape: # (T encoder, N, EMBEDDING_DIM)

        encoder_states, (hidden, cell) = self.rnn(embedding)
        #  encoder_states shape:  (T encoder *num_layers , N, 2*hidden_size)
        # hidden, cell : (2*num_layers, N, hidden_size) bidirectional = True

        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))
        # now : hidden, cell : (num_layers, N, hidden_size)

        return encoder_states, hidden, cell


class Decoder(nn.Module):
    def __init__(
        self, input_size, embedding_size, hidden_size, output_size, num_layers, p
    ):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size  # LATENT_DIM
        self.num_layers = num_layers  # 1 or 2

        self.embedding = nn.Embedding(
            input_size, embedding_size
        )  # input size = vocab fr size = num_words_output
        self.rnn = nn.LSTM(
            hidden_size * 2 + embedding_size, hidden_size, num_layers
        )  # ), dropout=p)
        # -> T decoder x N x hidden

        self.energy = nn.Linear(hidden_size * 3, 1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)  # ->  T x N x output size

    def forward(self, x, encoder_states, hidden, cell):
        # x shape: (N) where N is for batch size, we want it to be (N, 1), seq_length
        # is 1 here because we are sending in a single word and not a sentence
        x = x.unsqueeze(0)  # -> (1, N)

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)

        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        # (sequence_length * num_layers, N, hidden_size)

        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        # -> (sequence_length, N, 1)
        attention = self.softmax(energy)
        # (sequence_length, N, 1)
        attention = attention.permute(1, 2, 0)
        # (N, 1, sequence_length)
        encoder_states = encoder_states.permute(1, 0, 2)
        # (N, T, 2*hidden)

        context_vector = torch.bmm(attention, encoder_states).permute(1, 0, 2)
        # (N, 1, 2*hidden) -> (1, N, 2*hidden)

        rnn_input = torch.cat((context_vector, embedding), dim=2)
        # rnn_input: (1, N, hidden_size*2 + embedding_size)

        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # (1, N, 2*hidden + embedding_size) ->(1, N,  hidden)
        # outputs shape: (1, N,  hidden)
        # hidden, cell: (1, N,  hidden)

        predictions = self.fc(outputs)
        # -> (1, N, output_size)

        # predictions shape: (N, length_target_vocabulary) to send it to
        # loss function we want it to be (N, length_target_vocabulary) so we're
        # just gonna remove the first dim
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell
        #


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, teacher_force_ratio=1):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_force_ratio = teacher_force_ratio

    def forward(self, source, target):
        # source = encoder_inputs
        # target = encoder_inputs
        batch_size = source.shape[1]  # source (T_encoder, N)
        target_len = target.shape[0]  # target (T_decoder, N)
        target_vocab_size = len(
            word2idx_eng
        )  # check this is correct num_words = len(word2idx_outputs) + 1

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(
            device
        )  # (T_decoder, N, vocab)

        encoder_states, hidden, cell = self.encoder(
            source
        )  # (num_layers, N, hidden_size)

        # Grab the first input to the Decoder which will be <SOS> token
        x = target[0]  # (1, N)
        outputs[-1] = word2idx_eng[eos] * torch.ones(batch_size, target_vocab_size).to(
            device
        )

        for t in range(1, target_len):
            # Use previous hidden, cell as context from encoder at start
            # hidden, cell = hidden.squeeze(1), cell.squeeze(1)
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)

            # Store next output prediction
            outputs[t - 1] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the Decoder predicted it to be.
            # Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing is 1
            # then inputs at test time might be completely different than what the
            # network is used to. This was a long comment.
            x = target[t] if random.random() < self.teacher_force_ratio else best_guess

        return outputs
