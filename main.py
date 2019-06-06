# coding: utf-8
import argparse
import time
import datetime
import math
import os
import torch
import torch.nn as nn
import torchtext

import dataset
import model

parser = argparse.ArgumentParser(description='text classification')
parser.add_argument('--data', type=str, default='',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='cuda')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--note', type=str, default="",
                    help='extra note in final one-line result output')
parser.add_argument('--encoder', type=str, default="rnn",
                    help='rnn, avg')
parser.add_argument('--bidirectional', action='store_true',
                    help='help')
args = parser.parse_args()
print(args)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

device = torch.device(args.device)

#######################
#  Build the dataset  #
#######################

TEXT, LABEL, train_iter, val_iter, test_iter = dataset.create_data_iter(
    args.batch_size,
    device,
    args.data
)

################################
#  CNN highway network config  #
################################
class ConfigCNN(object):
    def __init__(self, arg=None):
        super(ConfigCNN, self).__init__()
        self.char_dim_cnn = args.emsize
        self.char_conv_fn = [100, 50, 50, 40]
        self.char_conv_fh = [1] * len(self.char_conv_fn)
        self.char_conv_fw = [1, 2, 3, 4]
        # setting for short text
        #  self.char_conv_fn = [100]
        #  self.char_conv_fh = [1] * len(self.char_conv_fn)
        #  self.char_conv_fw = [1]

config = ConfigCNN()


###############################################################################
# Build the model
###############################################################################

label_num = len(LABEL.vocab.itos)
vocab_size = len(TEXT.vocab.itos)
padding_index = TEXT.vocab.stoi["<pad>"]
print(f"vocab_size: {vocab_size} label_num: {label_num}")

model = model.RNN(
    vocab_size,
    args.emsize,
    label_num,
    padding_index=padding_index,
    hidden_size=args.nhid,
    num_layers=args.nlayers,
    encoder_model=args.encoder,
    dropout_p=args.dropout,
    bidirectional=args.bidirectional,
    cnn_config=config
).to(device)


optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr,
    weight_decay=args.weight_decay
)
criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def evaluate(data_iter):
    model.eval()
    correct = 0.
    example_num = 0
    with torch.no_grad():
        for counter, batch in enumerate(data_iter, 1):
            text, text_len = batch.text[0], batch.text[1]
            label = batch.label

            output = model(text, text_len)
            _, predicted_label = output.view(-1, label_num).max(dim=1)
            correct += (predicted_label == label).sum().item()
            example_num += len(label)

    return correct / example_num


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    for counter, batch in enumerate(train_iter, 1):
        text, text_len = batch.text[0], batch.text[1]
        label = batch.label

        #  model.zero_grad()
        optimizer.zero_grad()
        output = model(text, text_len)
        loss = criterion(output, label)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        #  for p in model.parameters():
            #  if p.grad is not None:
                #  p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()
    return total_loss


# Loop over epochs.
lr = args.lr
best_val_correct_rate = None

# At any point you can hit Ctrl + C to break out of training early.
all_train_start_time = datetime.datetime.now()
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_loss = train()
        correct_rate = evaluate(test_iter)

        print(f"epoch {epoch} | time {time.time() - epoch_start_time:.1f}s | train loss {train_loss} | correct rate on test {correct_rate} | lr {lr}")
        if not best_val_correct_rate or correct_rate > best_val_correct_rate:
            best_val_correct_rate = correct_rate
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            #  lr /= 2.0
            pass
        if epoch % 4 == 0:
            pass
            #  test_correct_rate = evaluate(test_iter)
            #  print(f"correct rate on test: {test_correct_rate}")
        #  if lr < 0.05:
            #  break
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

all_train_end_time = datetime.datetime.now()
all_train_time = f"{all_train_end_time - all_train_start_time}"
print(f"all training time elapsed: {all_train_time}")
print(f"training time per epoch: {(all_train_end_time - all_train_start_time) / args.epochs}")


# Run on test data.
test_correct_rate = evaluate(test_iter)
print(f"correct rate on test: {test_correct_rate}")
