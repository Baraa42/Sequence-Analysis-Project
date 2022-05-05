import torch
from torch.utils import data


# build data set
class Translation(data.Dataset):
    def __init__(
        self,
        max_len_input,
        max_len_target,
        input_sequences,
        target_sequences,
        target_sequences_inputs,
    ):
        self.max_len_input = max_len_input
        self.max_len_target = max_len_target
        self.input_sequences = input_sequences
        self.target_sequences = target_sequences
        self.target_sequences_inputs = target_sequences_inputs

    def __len__(self):
        return len(self.input_sequences)

    def pad_input(self, sequence):
        max_len = self.max_len_input
        sequence_len = len(sequence)
        if sequence_len >= max_len:
            sequence = sequence[:max_len]
        else:
            sequence = [0] * (max_len - sequence_len) + sequence

        return sequence

    def pad_target(self, sequence):
        max_len = self.max_len_target
        sequence_len = len(sequence)
        if sequence_len >= max_len:
            sequence = sequence[:max_len]
        else:
            sequence = sequence + [0] * (max_len - sequence_len)
        return sequence

    def __getitem__(self, idx):

        encoder_in = self.pad_input(self.input_sequences[idx])
        decoder_in = self.pad_target(self.target_sequences_inputs[idx])
        decoder_out = self.pad_target(self.target_sequences[idx])

        encoder_in = torch.LongTensor(encoder_in)
        decoder_in = torch.LongTensor(decoder_in)
        decoder_out = torch.FloatTensor(decoder_out)

        return encoder_in, decoder_in, decoder_out


# build data set
class TwitterDatset(data.Dataset):
    def __init__(
        self,
        max_len_input,
        max_len_target,
        input_sequences,
        target_sequences,
        target_sequences_inputs,
    ):
        self.max_len_input = max_len_input
        self.max_len_target = max_len_target
        self.input_sequences = input_sequences
        self.target_sequences = target_sequences
        self.target_sequences_inputs = target_sequences_inputs

    def __len__(self):
        return len(self.input_sequences)

    def pad_input(self, sequence):
        max_len = self.max_len_input
        sequence_len = len(sequence)
        if sequence_len >= max_len:
            sequence = sequence[:max_len]
        else:
            sequence = [0] * (max_len - sequence_len) + sequence

        return sequence

    def pad_target(self, sequence):
        max_len = self.max_len_target
        sequence_len = len(sequence)
        if sequence_len >= max_len:
            sequence = sequence[:max_len]
        else:
            sequence = sequence + [0] * (max_len - sequence_len)
        return sequence

    def __getitem__(self, idx):

        encoder_in = self.pad_input(self.input_sequences[idx])
        decoder_in = self.pad_target(self.target_sequences_inputs[idx])
        decoder_out = self.pad_target(self.target_sequences[idx])

        encoder_in = torch.LongTensor(encoder_in)
        decoder_in = torch.LongTensor(decoder_in)
        decoder_out = torch.FloatTensor(decoder_out)

        return encoder_in, decoder_in, decoder_out
