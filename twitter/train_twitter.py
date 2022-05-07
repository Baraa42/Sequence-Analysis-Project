import nltk
import nltk.translate.bleu_score as bleu
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

from utils import unicodeToAscii, normalizeString, build_vocab, build_sequences
from datasets import dataset
from model import model

nltk.download("punkt")


# from google.colab import drive
# drive.mount('/content/drive')

drive_path = "/content/drive/MyDrive/Sequence-Analysis-Projects"
twitter_path = os.path.join(drive_path, "twitter-chatbot")
if not os.path.exists(drive_path):
    os.mkdir(drive_path)

projects = ["translation", "twitter-chatbot"]
for project in projects:
    project_path = os.path.join(drive_path, project)
    if not os.path.exists(project_path):
        os.mkdir(project_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# some configuration
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20_000
NUM_SAMPLES = 10_000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 64
EPOCHS = 100
LATENT_DIM = 512

bos = "<bos>"
eos = "<eos>"
pad = "<pad>"
unk = "<unk>"


# load in the data
input_texts = []  # sentence in original language
target_texts = []  # sentence in target language
# target_text_inputs = []  #  sentence in target language offset by 1
t = 0
for line in open(os.path.join(twitter_path, "twitter_tab_format.txt")):
    # only keep a limited number of samples
    t += 1
    if t > NUM_SAMPLES:
        break
    line = line.rstrip()
    # input and targets are separeted by tab
    if "\t" not in line:
        continue

    # split up the input and translation
    input_text, target_text = line.split("\t")[:2]

    input_text, target_text = normalizeString(input_text), normalizeString(target_text)

    # target_text_input = "<sos> " + translation

    input_texts.append(input_text)
    target_texts.append(target_text)
    # target_text_inputs.append(target_text_input)
print("num samples:", len(input_texts))

english_vocab = build_vocab(input_texts)
# english_vocab = build_vocab(train_inputs)
print(f"There is {len(english_vocab)} token in english_vocab")
english_vocab = [pad, unk, bos, eos] + english_vocab

word2idx_eng = {}
for i in range(len(english_vocab)):
    word2idx_eng[english_vocab[i]] = i

# map indexes back into real words
# so wwe can view the results
idx2word_eng = {v: w for w, v in word2idx_eng.items()}


### Build Train/Test datasets
N = len(input_texts)
indices = np.random.permutation(N)

N_train = 4 * N // 5

train_indices = list(indices[:N_train])
test_indices = list(indices[N_train:])

train_inputs = [input_texts[i] for i in train_indices]
train_targets = [target_texts[i] for i in train_indices]

test_inputs = [input_texts[i] for i in test_indices]
test_targets = [target_texts[i] for i in test_indices]


print("len(train_inputs):", len(train_inputs))
print("len(test_inputs):", len(test_inputs))

input_sequences_train = build_sequences(train_inputs, word2idx_eng)
target_sequences_train = build_sequences(train_targets, word2idx_eng)

input_sequences_test = build_sequences(test_inputs, word2idx_eng)
target_sequences_test = build_sequences(test_targets, word2idx_eng)

print("len(input_sequences_train):", len(input_sequences_train))
print("len(input_sequences_test):", len(input_sequences_test))


bos_idx = word2idx_eng[bos]
eos_idx = word2idx_eng[eos]
# input_sequences = [[bos_idx] + sequence + [eos_idx] for sequence in input_sequences]
# target_sequences_inputs = [[bos_idx] + sequence for sequence in target_sequences]
# target_sequences = [sequence + [eos_idx] for sequence in target_sequences]

target_sequences_train_inputs = [
    [bos_idx] + sequence for sequence in target_sequences_train
]
target_sequences_train = [sequence + [eos_idx] for sequence in target_sequences_train]


target_sequences_test_inputs = [
    [bos_idx] + sequence for sequence in target_sequences_test
]
target_sequences_test = [sequence + [eos_idx] for sequence in target_sequences_test]

# max_len_input = max(len(s) for s in input_sequences)
max_len_input = max(len(s) for s in input_sequences_train)

print("max_len_input:", max_len_input)

max_len_target = max(len(s) for s in target_sequences_train)
# max_len_target = max(len(s) for s in target_sequences)

print("max_len_target:", max_len_target)


# load in pre-trained word vectors
print("loading word vectors ...")
word2vec_path = "glove.6B.%sd.txt"
word2vec = {}
with open(os.path.join(word2vec_path % EMBEDDING_DIM)) as f:
    # is just a space-separated text file in the format:
    # word vec[0] vec[1] vec[2]
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.array(values[1:], dtype="float32")
        word2vec[word] = vec
    print("Found %s word vectors." % len(word2vec))


# prepare embedding matrix
print("Filling pre-trained embeddings...")
num_words = min(MAX_NUM_WORDS, len(word2idx_eng))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx_eng.items():
    embedding_vector = word2vec.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all zeros
        embedding_matrix[i] = embedding_vector


# twitter_dataset = TwitterDatset(max_len_input, max_len_target, input_sequences, target_sequences, target_sequences_inputs)

train_dataset = dataset.TwitterDatset(
    max_len_input,
    max_len_target,
    input_sequences_train,
    target_sequences_train,
    target_sequences_train_inputs,
)
test_dataset = dataset.TwitterDatset(
    max_len_input,
    max_len_target,
    input_sequences_test,
    target_sequences_test,
    target_sequences_test_inputs,
)


# instantiate dataloaders
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, shuffle=True, batch_size=BATCH_SIZE
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, shuffle=False, batch_size=BATCH_SIZE
)


# load pre-trained word embeddings into an embedding layer
# freeze the layer
embedding_layer = nn.Embedding(num_words, EMBEDDING_DIM)  # vocab size  # embedding dim
embedding_layer.weight = nn.Parameter(torch.from_numpy(embedding_matrix).float())
embedding_layer.requires_grad = False


input_size_encoder = num_words
input_size_decoder = num_words
output_size = num_words
encoder_embedding_size = EMBEDDING_DIM
decoder_embedding_size = EMBEDDING_DIM
hidden_size = LATENT_DIM  # Needs to be the same for both RNN's
num_layers = 1
enc_dropout = 0.5
dec_dropout = 0.5
teacher_force_ratio = 0.5
# Training hyperparameters
num_epochs = 300
learning_rate = 0.001
batch_size = 64


# instantiate models
encoder_net = model.Encoder(
    embedding_layer, encoder_embedding_size, hidden_size, num_layers, enc_dropout
).to(device)

decoder_net = model.Decoder(
    input_size_decoder,
    decoder_embedding_size,
    hidden_size,
    output_size,
    num_layers,
    dec_dropout,
).to(device)

model = model.Seq2Seq(encoder_net, decoder_net).to(device)

# if loading
# model = torch.load('./twitter_chatbot.pth')

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate
)  # , weight_decay=1e-3)
model.train()


num_epochs = 100

train_losses = np.zeros(num_epochs)
test_losses = np.zeros(num_epochs)

for epoch in range(num_epochs):
    print(f"[Epoch {epoch+1} / {num_epochs}]")

    model.train()
    for batch_idx, batch in enumerate(train_loader):
        train_loss = []
        # Get input and targets and get to cuda
        # encoder_in, decoder_in, decoder_out
        encoder_in, decoder_in, target = batch
        encoder_in, decoder_in, target = (
            encoder_in.to(device),
            decoder_in.to(device),
            target.to(device),
        )
        encoder_in, decoder_in, target = (
            encoder_in.permute(1, 0),
            decoder_in.permute(1, 0),
            target.permute(1, 0),
        )

        # Forward prop
        output = model(encoder_in, decoder_in)

        # Output is of shape (batch_size, trg_len, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have batch_size * output_words that we want to send in into
        # our cost function, so we need to do some reshapin. While we're at it
        # Let's also remove the start token while we're at it
        # output = output[1:].reshape(-1, output.shape[2])
        # target = target[1:].reshape(-1)
        output = output.reshape(-1, output.shape[2])
        target = target.reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target.long())

        # Back prop
        loss.backward()

        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()
        train_loss.append(loss.item())

    model.eval()
    for batch_idx, batch in enumerate(test_loader):
        test_loss = []
        # Get input and targets and get to cuda
        # encoder_in, decoder_in, decoder_out
        encoder_in, decoder_in, target = batch
        encoder_in, decoder_in, target = (
            encoder_in.to(device),
            decoder_in.to(device),
            target.to(device),
        )
        encoder_in, decoder_in, target = (
            encoder_in.permute(1, 0),
            decoder_in.permute(1, 0),
            target.permute(1, 0),
        )

        # Forward prop
        output = model(encoder_in, decoder_in)

        output = output.reshape(-1, output.shape[2])
        target = target.reshape(-1)
        loss = criterion(output, target.long())
        test_loss.append(loss.item())

    epoch_train_loss = np.mean(train_loss)
    train_losses[epoch] = epoch_train_loss

    epoch_test_loss = np.mean(test_loss)
    test_losses[epoch] = epoch_test_loss

    print(f"Train Loss: {epoch_train_loss:.3f}, Test Loss: {epoch_test_loss:.3f}")


# some plots
plt.plot(train_losses, label="train_loss")
plt.plot(test_losses, label="test_loss")
plt.legend()
plt.show()

torch.save(model, os.path.join(twitter_path, f"twitter_chatbot.pth"))

model.eval()


softmax = nn.Softmax(dim=1)


def decode_sequence(input_seq, mode="greedy"):
    # encode the input as state vectors.
    input_seq = input_seq.to(device)
    with torch.no_grad():
        encoder_states, h, c = encoder_net(input_seq)

        # generate empty target seq of length 1
        target_seq = torch.zeros(1).int().to(device)

        # populate the first character of target sequence with the start character
        # NOTE: tokenizer lower cases all words
        target_seq[0] = word2idx_eng[bos]

        # if we get this we break
        eos_idx = word2idx_eng[eos]

        # create translation
        output_sentence = []
        for _ in range(max_len_target):
            output_tokens, h, c = decoder_net(target_seq, encoder_states, h, c)

            if mode == "sample":
                probs = softmax(output_tokens).view(-1)
                idx = np.random.choice(len(probs), p=probs.detach().cpu().numpy())

            else:
                # get next word
                idx = output_tokens.argmax(1).item()

            # end of sentence EOS
            if eos_idx == idx:
                break

            word = ""
            if idx > 0:
                word = idx2word_eng[idx]
                output_sentence.append(word)

            # update the decoder input
            # which is just the word just generated
            target_seq[0] = idx
            # states_value = [h, c]

        return " ".join(output_sentence)


encoder_net = model.encoder
decoder_net = model.decoder
while True:
    # do some answering
    i = np.random.choice(len(train_dataset))
    input_seq, _, _ = train_dataset[i]
    input_seq = input_seq.unsqueeze(1)
    pred_answer = decode_sequence(input_seq, mode="sample")
    true_answer = train_targets[i]
    print("_")
    print("Input:", train_inputs[i])
    print("True answer:", true_answer)  # [:-5])
    print("Predicted:", pred_answer)

    ans = input("Continue? [Y/n]")
    if ans and ans.lower().startswith("n"):
        break


while True:
    # do some answering
    i = np.random.choice(len(test_dataset))
    input_seq, _, _ = test_dataset[i]
    input_seq = input_seq.unsqueeze(1)
    pred_answer = decode_sequence(input_seq, mode="sample")
    true_answer = test_targets[i]
    print("_")
    print("Input:", test_inputs[i])
    print("True answer:", true_answer)  # [:-5])
    print("Predicted:", pred_answer)

    ans = input("Continue? [Y/n]")
    if ans and ans.lower().startswith("n"):
        break


def build_sequences_tweet(tweet, word2idx):
    tweet_tokenized = word_tokenize(tweet)
    tweet_seq = []
    unk_idx = word2idx_eng[unk]
    for word in tweet_tokenized:
        tweet_seq.append(word2idx_eng.get(word, unk_idx))

    return tweet_seq


def answer_tweet(tweet, mode="greedy"):
    tweet_seq = build_sequences_tweet(tweet, word2idx_eng)
    tweet_seq = torch.LongTensor(tweet_seq).unsqueeze(1)
    return decode_sequence(tweet_seq, mode)


tweet = "im going to watch some football tonight and eat chinese"
answer_tweet(tweet, "sample")
