import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNN_Text(nn.Module):

    def __init__(self, embedding, params):

        super(CNN_Text, self).__init__()

        self.args = params
        n_words = embedding.shape[0]
        emb_dim = embedding.shape[1]
        filters = params.filters

        if params.kernels <= 3:
            Ks = [i for i in range(1, params.kernels + 1)]
        else:
            Ks = [i for i in range(1, params.kernels + 1, 2)]
        C = params.n_out

        self.embed = nn.Embedding(n_words, emb_dim)
        self.embed.weight = nn.Parameter(torch.from_numpy(embedding), requires_grad=False)

        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, filters, K) for K in Ks])

        self.fc = nn.Linear(len(Ks) * filters, C)

        self.dropout = nn.Dropout(p=self.args.dropout)
        self.embed_dropout = nn.Dropout(p=self.args.embed_dropout)

    def fc_layer(self, x, layer):
        x = layer(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        return x

    def encoder(self, x):
        x = self.embed(x)
        x = x.transpose(1, 2)
        x = self.embed_dropout(x)

        h = [self.fc_layer(x, self.convs[i]) for i in range(len(self.convs))]

        return h

    def forward(self, Note):
        h = self.encoder(Note)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in h]

        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return torch.sigmoid(x)


class CNN_Text_Attn(nn.Module):
    def __init__(self, embedding, params):

        super(CNN_Text_Attn, self).__init__()

        self.args = params
        n_words = embedding.shape[0]
        emb_dim = embedding.shape[1]
        filters = params.filters

        if (params.kernels <= 3):
            Ks = [i for i in range(1, params.kernels + 1)]
        else:
            Ks = [i for i in range(1, params.kernels + 1, 2)]
        C = params.n_out

        self.embed = nn.Embedding(n_words, emb_dim)
        self.embed.weight = nn.Parameter(torch.from_numpy(embedding), requires_grad=False)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, filters, K) for K in Ks])

        self.U = nn.Linear(filters, 1, bias=False)

        self.fc = nn.Linear(filters, C)

        self.dropout = nn.Dropout(p=self.args.dropout)
        self.embed_dropout = nn.Dropout(p=self.args.embed_dropout)
        self.attn_dropout = nn.Dropout(p=self.args.attn_dropout)

        self.padding = nn.ModuleList([nn.ConstantPad1d((0, K - 1), 0) for K in Ks])


    def fc_layer(self, x, layer, padding):
        x = layer(padding(x))
        x = F.leaky_relu(x)
        x = self.dropout(x)

        return x

    def encoder(self, x):
        x = self.embed(x)
        x = self.embed_dropout(x)
        x = x.transpose(1, 2)
        h = [self.fc_layer(x, self.convs[i], self.padding[i]) for i in range(len(self.convs))]

        return h

    def forward(self, Note, interpret=False):

        text = Note[0]
        attn_mask = torch.cat([Note[1]] * len(self.convs), 1)
        h = self.encoder(text)

        h = torch.cat(h, 2).transpose(1, 2)

        alpha = torch.add(self.U(h / (self.args.filters)**0.5), attn_mask.unsqueeze(2))
        h = h.transpose(1, 2).matmul(F.softmax(alpha, dim=1)).squeeze(2)

        h = self.dropout(h)
        y_hat = self.fc(h)

        if interpret:
            return torch.sigmoid(y_hat), [alpha.squeeze()]
        else:
            return torch.sigmoid(y_hat)


class CNN_Text_Attn_Phen(nn.Module):
    def __init__(self, embedding, params):

        super(CNN_Text_Attn_Phen, self).__init__()

        self.args = params
        n_words = embedding.shape[0]
        emb_dim = embedding.shape[1]
        filters = params.filters

        if (params.kernels <= 3):
            Ks = [i for i in range(1, params.kernels + 1)]
        else:
            Ks = [i for i in range(1, params.kernels + 1, 2)]

        self.embed = nn.Embedding(n_words, emb_dim)
        self.embed.weight = nn.Parameter(torch.from_numpy(embedding), requires_grad=False)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, filters, K) for K in Ks])

        self.U = nn.Linear(filters, params.num_phenotypes, bias=False)

        self.phen_proj = nn.Linear(params.filters, params.num_phenotypes)
        self.final_proj = nn.Linear(params.num_phenotypes, 1)

        self.dropout = nn.Dropout(p=self.args.dropout)
        self.embed_dropout = nn.Dropout(p=self.args.embed_dropout)
        self.attn_dropout = nn.Dropout(p=self.args.attn_dropout)

        self.padding = nn.ModuleList([nn.ConstantPad1d((0, K - 1), 0) for K in Ks])
        self.frozen = False

    def fc_layer(self, x, layer, padding):
        x = layer(padding(x))
        x = F.leaky_relu(x)
        x = self.dropout(x)

        return x

    def encoder(self, x):
        x = self.embed(x)
        x = self.embed_dropout(x)
        x = x.transpose(1, 2)
        h = [self.fc_layer(x, self.convs[i], self.padding[i]) for i in range(len(self.convs))]

        return h

    def freeze(self):
        self.dropout = nn.Dropout(p=0)
        self.embed_dropout = nn.Dropout(p=0)
        for conv in self.convs:
            for param in conv.parameters():
                param.requires_grad = False
        for param in self.U.parameters():
            param.requires_grad = False
        for param in self.phen_proj.parameters():
            param.requires_grad = False
        self.frozen = True
        self.args.task == 'other'

    def forward(self, Note, interpret=False):

        text = Note[0]
        attn_mask = torch.cat([Note[1]] * len(self.convs), 1)
        h = self.encoder(text)

        h = torch.cat(h, 2).transpose(1, 2)

        alpha = F.softmax(torch.add(self.U(h / (self.args.filters) ** 0.5), attn_mask.unsqueeze(2)), dim=1)
        h = h.transpose(1, 2).matmul(alpha).squeeze(2)

        phen_scores = self.phen_proj.weight.mul(h.transpose(1, 2)).sum(dim=2).add(self.phen_proj.bias)

        phen_scores = self.dropout(phen_scores)

        y_hat = self.final_proj(phen_scores)

        if interpret:
            contribution = self.final_proj.weight.mul(phen_scores)
            alpha = torch.unbind(alpha.squeeze(), dim=2)

            return torch.sigmoid(y_hat), alpha, torch.sigmoid(phen_scores), contribution
        else:
            if self.args.task == 'icd_only':
                return torch.sigmoid(phen_scores)
            elif 'allied' in self.args.task:
                return torch.sigmoid(y_hat), torch.sigmoid(phen_scores)
            else:
                return torch.sigmoid(y_hat)


class lr_baseline(nn.Module):
    def __init__(self, params):

        super(lr_baseline, self).__init__()
        self.proj = nn.Linear(params.num_phenotypes, 1, bias=True)

    def forward(self, codes):
        return torch.sigmoid(self.proj(codes))




class LSTM_Attn(nn.Module):
    def __init__(self, embedding, params):

        super(LSTM_Attn, self).__init__()
        self.args = params
        n_words = embedding.shape[0]
        emb_dim = embedding.shape[1]

        C = params.n_out
        h = params.h
        self.h = h
        self.embed = nn.Embedding(n_words, emb_dim)
        self.embed.weight = nn.Parameter(torch.from_numpy(embedding), requires_grad=False)

        self.rnn = nn.LSTM(input_size=emb_dim, hidden_size=h, batch_first=True, bidirectional=True)

        self.attn1 = nn.Linear(h*2, h)
        self.attn2 = nn.Linear(h, 1, bias=False)

        self.dropout = nn.Dropout(p=self.args.dropout)
        self.embed_dropout = nn.Dropout(p=self.args.embed_dropout)

        if params.bidir:
            self.fc = nn.Linear(h*2, C)
        else:
            self.fc = nn.Linear(h, C)

    def forward(self, Note):

        text = Note[0]
        attn_mask = Note[1]

        x = self.embed_dropout(self.embed(text))

        output, (_, _) = self.rnn(x)


        attn1 = nn.Tanh()(self.attn1(output))
        attn2 = self.attn2(attn1)
        attn = F.softmax(torch.add(attn2, attn_mask.unsqueeze(2)), dim=1)

        h = output.transpose(1, 2).matmul(attn).squeeze(2)

        h = self.dropout(h)
        y_hat = self.fc(h)
        return torch.sigmoid(y_hat)


def accuracy(outputs, labels):
    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.ravel()

    # np.argmax gives us the class predicted for each token by the model
    outputs = outputs.ravel()
    outputs[outputs >= .5] = 1
    outputs[outputs < .5] = 0

    # compare outputs with labels and divide by number of tokens (excluding PADding tokens)
    return np.sum(outputs == labels) / float(outputs.size)


metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
