import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN(nn.Module):

    def __init__(self, vocab_size, embed_size, num_output, rnn_model='LSTM', use_last=False, embedding_tensor=None,
                 padding_index=0, hidden_size=64, num_layers=1, batch_first=True,
                 encoder_model="rnn", dropout_p=0):
        """

        Args:
            vocab_size: vocab size
            embed_size: embedding size
            num_output: number of output (classes)
            rnn_model:  LSTM or GRU
            use_last:  bool
            embedding_tensor:
            padding_index:
            hidden_size: hidden size of rnn module
            num_layers:  number of layers in rnn module
            batch_first: batch first option
        """

        super(RNN, self).__init__()
        self.use_last = use_last
        self.encoder_model = encoder_model
        self.hidden_size = hidden_size
        # embedding
        self.encoder = None
        if torch.is_tensor(embedding_tensor):
            self.encoder = nn.Embedding(vocab_size, embed_size, padding_idx=padding_index, _weight=embedding_tensor)
            self.encoder.weight.requires_grad = False
        else:
            self.encoder = nn.Embedding(vocab_size, embed_size, padding_idx=padding_index)

        #  self.drop_en = nn.Dropout(p=0.6)
        self.dropout_p = dropout_p
        self.drop_en = nn.Dropout(self.dropout_p)

        # rnn module
        if rnn_model == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=self.dropout_p,
                batch_first=True,
                bidirectional=True
            )
        elif rnn_model == 'GRU':
            self.rnn = nn.GRU(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=self.dropout_p,
                batch_first=True,
                bidirectional=True
            )
        else:
            raise LookupError(' only support LSTM and GRU')


        #  out_hidden_size = hidden_size * 2
        out_hidden_size = hidden_size
        if self.encoder_model == "avg":
            out_hidden_size = embed_size
        #  self.bn2 = nn.BatchNorm1d(out_hidden_size)
        self.fc = nn.Linear(out_hidden_size, num_output)

    def forward_avg(self, x, seq_lengths):
        '''
        Args:
            x: (batch, time_step, input_size)

        Returns:
            num_output size
        '''

        x_embed = self.encoder(x)
        # shape: batch_size, seq_len, embed_size
        x_embed = self.drop_en(x_embed)
        output = x_embed.sum(dim=1)
        output = output / seq_lengths.type(output.dtype).unsqueeze(1).expand(output.shape)
        #  fc_input = self.bn2(output)
        # shape: (batch_size, label_num)
        out = self.fc(output)
        return out

    def forward(self, x, seq_lengths):
        if self.encoder_model == "avg":
            return self.forward_avg(x, seq_lengths)
        if self.encoder_model == "rnn":
            return self.forward_rnn(x, seq_lengths)


    def forward_rnn(self, x, seq_lengths):
        '''
        Args:
            x: (batch, time_step, input_size)

        Returns:
            num_output size
        '''

        x_embed = self.encoder(x)
        x_embed = self.drop_en(x_embed)
        packed_input = pack_padded_sequence(x_embed, seq_lengths.cpu().numpy(),batch_first=True)

        # out_rnn shape (batch, seq_len, hidden_size * num_directions)
        # None is for initial hidden state
        packed_output, ht = self.rnn(packed_input, None)
        out_rnn, _ = pad_packed_sequence(packed_output, batch_first=True)

        row_indices = torch.arange(0, x.size(0)).long()
        col_indices = seq_lengths - 1
        if next(self.parameters()).is_cuda:
            row_indices = row_indices.cuda()
            col_indices = col_indices.cuda()

        if self.use_last:
            last_tensor=out_rnn[row_indices, col_indices, :]
        else:
            # use mean
            batch_size, seq_len, out_size = out_rnn.shape

            # forward + backward
            bilstm_out = out_rnn.view(batch_size, seq_len, 2, self.hidden_size)
            bilstm_out = bilstm_out[:, :, 0, :] + bilstm_out[:, :, 1, :]
            # batch_size, hidden_size
            bilstm_out = torch.sum(bilstm_out, dim=1)
            last_tensor = bilstm_out / seq_lengths.type(bilstm_out.dtype).unsqueeze(1).expand(bilstm_out.shape)
            #  last_tensor = out_rnn[row_indices, :, :]
            #  last_tensor = torch.mean(last_tensor, dim=1)

        #  fc_input = self.bn2(last_tensor)
        # shape: (batch_size, label_num)
        out = self.fc(last_tensor)
        return out
