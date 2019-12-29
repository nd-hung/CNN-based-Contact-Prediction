import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

RAW_CHANNELS = 441

# amino acid indices
aa2ix = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
         'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}


def aanum(ch):
    if ch in aa2ix:
        return aa2ix[ch]
    else:
        return 20


def encode_sequence(sequence):
    idxs = [aanum(w) for w in sequence]
    return torch.tensor(idxs, dtype=torch.long)


# define conv layer block for ResNet-based models
def conv3x3(in_channels, out_channels, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=padding, dilation=dilation)


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0)


def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5,
                     stride=stride, padding=2, bias=False)


def get_output(x, length):
    p2 = x[0, 0:length, 0:length]
    p3 = (p2 + p2.transpose(1, 0)) / 2
    p3 = p3.reshape(1, length, length)
    return p3


# Bottleneck block for DeepCon model
class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 padding=1, dilation=1, dropout=0.3):
        super(BottleNeck, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv3x3(out_channels, out_channels, padding=padding,
                             dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv2.weight,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out


# basic block for DeepCov model
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=5, stride=1, padding=2, bias=False)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv.weight,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class DeepCov(nn.Module):
    def __init__(self, block, layers, in_channels=441):
        super(DeepCov, self).__init__()
        self.in_channels = in_channels
        self.conv1x1 = nn.Conv2d(441, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(1)
        self.maxpool2d = nn.MaxPool3d((2, 1, 1))
        self.conv5x5 = nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=False)
        self.last_conv = nn.Conv2d(64, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.layer1 = self.make_layers(block, 64, layers[0])
        self.layer2 = self.make_layers(block, 64, layers[1])
        self.layer3 = self.make_layers(block, 64, layers[2])
        self.layer4 = self.make_layers(block, 64, layers[3])
        self.layer5 = self.make_layers(block, 64, layers[4])
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1x1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv5x5.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.last_conv.weight, gain=nn.init.calculate_gain('relu'))

    def make_layers(self, block, out_channels, blocks):
        layers = []
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # maxout layer
        out = self.conv1x1(x)
        out = self.bn1(out)
        out = self.maxpool2d(out)

        # feed max-out layer output through several conv layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        # the last layer needs to be of kernel size (1,1) and sigmoid activation
        out = self.last_conv(out)
        out = self.bn2(out)
        out = torch.sigmoid(out)

        return out


class DeepCon(nn.Module):
    def __init__(self, block, layers, in_channels=441):
        super(DeepCon, self).__init__()
        self.in_channels = in_channels
        self.conv1 = conv1x1(self.in_channels, 128)
        self.maxpool3d = nn.MaxPool3d(kernel_size=(2, 1, 1))
        self.bn1 = nn.BatchNorm2d(441)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 64, layers[1])
        self.conv_last = conv3x3(64, 1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv_last.weight,
                                gain=nn.init.calculate_gain('relu'))

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(64, out_channels, stride))
        self.in_channels = out_channels
        d_rate = 1
        for i in range(blocks):
            layers.append(block(out_channels, out_channels))
            layers.append(block(out_channels, out_channels, dilation=d_rate, padding=d_rate))
            if d_rate == 1:
                d_rate = 2
            elif d_rate == 2:
                d_rate = 4
            else:
                d_rate = 1
        return nn.Sequential(*layers)

    def forward(self, x):
        # maxout layer
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)  # (441, L, L) -> (128, L, L)
        out = self.maxpool3d(out)  # (128, L, L) -> (64, L, L)
        # end of maxout layer

        # residue layers
        out = self.layer1(out)  # (64, L, L) -> (64, L, L)
        out = self.layer2(out)

        # last layers
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv_last(out)  # (64, L, L) -> (1, L, L)
        out = torch.sigmoid(out)
        # return out
        return get_output(out[0], out.shape[-1])


# RNN-based contact prediction model
# Configuration init
class LSTMProtConfig:
    def __init__(self,
                 vocab_size=21,  # number of amino acids (20 + 1 unknown)
                 embed_dim=128,  # embedding size
                 hidden_dim=64,
                 num_layers=3,
                 dropout=0.1,
                 initializer_range=0.02):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.initializer_range = initializer_range


class ProtLSTMLayer(nn.Module):
    def __init__(self, config):
        super(ProtLSTMLayer, self).__init__()
        # create bi-directional LSTM model with batch first
        self.lstm = nn.LSTM(config.embed_dim, config.hidden_dim,
                            batch_first=True, num_layers=config.num_layers,
                            dropout=config.dropout, bidirectional=True)

    def forward(self, embedded_seq):  # batch first, e.g., (1, 70, 128)
        lstm_out = self.lstm(embedded_seq)
        # outputs = output, (h_n, c_n)
        # output shape: (seq_len, batch, num_directions * hidden_size)
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        return lstm_out


class ProtLSTMPooler(nn.Module):
    def __init__(self, config):
        super(ProtLSTMPooler, self).__init__()
        self.scalar_reweighting = nn.Linear(2 * config.num_layers, 1)
        self.dense = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        pooled_output = self.scalar_reweighting(hidden_states.permute(1, 2, 0)).squeeze(2)
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ProteinLSTMModel(nn.Module):
    def __init__(self, config):
        super(ProteinLSTMModel, self).__init__()
        # input of LSTM model is a protein sequence, so an embedding layer needed
        self.embed_matrix = nn.Embedding(config.vocab_size, config.embed_dim)
        self.encoder = ProtLSTMLayer(config)
        self.pooler = ProtLSTMPooler(config)
        self.config = config
        self.init_weights()

    def forward(self, input_ids):
        embedding_output = self.embed_matrix(input_ids).view(1, len(input_ids), -1)
        outputs = self.encoder(embedding_output)
        return outputs

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        self.apply(self._init_weights)



class PairwiseContactPredictionHead(nn.Module):
    """
    This module get the output of LSTM as its input and compute pairwise contact prediction.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.predict = nn.Sequential(
            nn.Dropout(), nn.Linear(4 * hidden_size, 1))

    def forward(self, inputs, targets=None):
        prod = inputs[:, :, None, :] * inputs[:, None, :, :]
        diff = inputs[:, :, None, :] - inputs[:, None, :, :]
        pairwise_features = torch.cat((prod, diff), -1)
        prediction = self.predict(pairwise_features)
        prediction = (prediction + prediction.transpose(1, 2)) / 2
        outputs = (prediction,)
        return outputs


class LSTMProt(nn.Module):
    def __init__(self, config):
        super(LSTMProt, self).__init__()
        self.lstm = ProteinLSTMModel(config)
        self.predict = PairwiseContactPredictionHead(config.hidden_dim)
        #self.init_weights()

    def forward(self, input_ids, targets=None):
        outputs = self.lstm(input_ids)
        sequence_output, pooled_output = outputs[:2]
        outputs = self.predict(sequence_output, targets) + outputs[2:]
        outputs = torch.sigmoid(outputs[0]).transpose(-1, 1)
        return outputs


class LSTMDeepCon(nn.Module):
    """
    This modules defines a combined LSTM-CNN model for contact prediction.
    The model gets two inputs and learns in parallel:
    - CNN-based model (DeepCon) accepts covariance matrices of amino acids (adopted from D. Jones).
    - LSTM-based model accepts protein sequences as input.
    Both the two branches return (L,L) contact prediction, where L is sequence length.
    """
    def __init__(self, config):
        super(LSTMDeepCon, self).__init__()
        self.config = config
        self.predict1 = deepcon()
        self.predict2 = LSTMProt(config)

    def forward(self, cov, aa_ids):
        cnn_out = self.predict1(cov)
        lstm_out = self.predict2(aa_ids)
        # return mean of the two outputs (ensemble)
        # or we can output the two separately then compute mean of the two BCE loss functions
        return (cnn_out + lstm_out) / 2.0


def deepcon():
    """
    This function creates a CNN-based, DEEPCON model for training contact prediction
    """
    net_args = {"block": BottleNeck, "layers": [8, 8]}
    model = DeepCon(**net_args)
    return model


def deepcov():
    """
    This function creates a CNN-based, DeepCov model for training contact prediction
    """
    net_args = {"block": BasicBlock, "layers": [2, 2, 2, 2, 2]}
    model = DeepCov(**net_args)
    return model


def make_lstm_contact_model():
    """
    This function creates a LSTM model for training contact prediction
    """
    config = LSTMProtConfig()
    model = LSTMProt(config)
    return model


def make_lstm_deepcon():
    """
    This function creates a LSTM-CNN combined model for training contact prediction
    """
    config = LSTMProtConfig()
    model = LSTMDeepCon(config)
    return model


