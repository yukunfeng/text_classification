#!/bin/sh

# python -u main.py --encoder "rnn" --data "data_2gram_no_oov" --emsize 50 --nhid 128 --nlayers 1 --epochs 30 --batch_size 128 --dropout 0.2 --lr 0.005
# python -u main.py --encoder "rnn" --data "data_2gram" --emsize 200 --nhid 200 --nlayers 1 --epochs 30 --batch_size 128 --dropout 0.2 --lr 0.005
# python -u main.py --encoder "rnn" --data "data" --emsize 200 --nhid 200 --nlayers 1 --epochs 30 --batch_size 128 --dropout 0.2 --lr 0.005
# python -u main.py --encoder "avg" --data "data_2gram" --emsize 200 --nhid 200 --nlayers 1 --epochs 30 --batch_size 128 --dropout 0.2 --lr 0.005
# python -u main.py --encoder "avg" --data "data" --emsize 200 --nhid 200 --nlayers 1 --epochs 30 --batch_size 128 --dropout 0.2 --lr 0.005

# python -u main.py --encoder "rnn" --bidirectional --data "data" --emsize 200 --nhid 200 --nlayers 1 --epochs 30 --batch_size 128 --dropout 0.2 --lr 0.005
# python -u main.py --encoder "rnn" --data "data" --emsize 200 --nhid 200 --nlayers 1 --epochs 30 --batch_size 128 --dropout 0.2 --lr 0.005
# python -u main.py --encoder "cnn" --data "data" --emsize 30 --nhid 128 --nlayers 1 --epochs 30 --batch_size 128 --dropout 0.5 --lr 0.005
# python -u main.py --encoder "cnn" --data "trec" --emsize 50 --nhid 128 --nlayers 1 --epochs 30 --batch_size 128 --dropout 0.5 --lr 0.005
python -u main.py --encoder "cnn" --bidirectional --data "sst2" --emsize 50 --nhid 128 --nlayers 1 --epochs 30 --batch_size 128 --dropout 0.5 --lr 0.005
