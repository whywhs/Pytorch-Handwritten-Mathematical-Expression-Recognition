'''
Python 3.6 
Pytorch 0.3
This project is produced by Hongyu Wang in June 2018 at MSRA.
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

batch_size = 1

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1,):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(self.dropout_p)

        self.embedding = nn.Embedding(self.output_size, batch_size * 256)
        self.gru = nn.GRUCell(684, 256)
        self.gru1 = nn.GRUCell(256, 256)
        self.out = nn.Linear(128, self.output_size)
        self.hidden = nn.Linear(256, 256)
        self.emb = nn.Linear(256, 128)
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.hidden2 = nn.Linear(256, 128)
        self.emb2 = nn.Linear(256, 128)
        self.ua = nn.Linear(684, 256)
        self.uf = nn.Linear(1, 256)
        self.v = nn.Linear(256, 1)
        self.wc = nn.Linear(684, 128)


    def forward(self, input, hidden, encoder_outputs,bb,attention_sum,decoder_attention,dense_input):

        # embedding the word from 1 to 256(total 112 words)
        embedded = self.embedding(input).view(batch_size,1,256)
        embedded = self.dropout(embedded)

        # st = GRU(y_t-1,s_t-1)
        st = self.gru1(embedded,hidden)
        hidden1 = self.hidden(st)

        # encoder_outputs from (1,height,width) => (height,width,1)
        encoder_outputs_trans = torch.transpose(encoder_outputs,0,1)
        encoder_outputs_trans = torch.transpose(encoder_outputs_trans,1,2)

        # encoder_outputs_trans (dense_input,bb,684) attention_sum_trans (dense_input,bb,1) hidden1 (1,1,256)
        decoder_attention = self.conv1(decoder_attention.unsqueeze(0))
        attention_sum = attention_sum + decoder_attention[0]
        attention_sum_trans = torch.transpose(attention_sum,0,1)
        attention_sum_trans = torch.transpose(attention_sum_trans,1,2)

        # encoder_outputs1 (dense_input,bb,256) attention_sum1 (dense_input,bb,256)
        encoder_outputs1 = self.ua(encoder_outputs_trans)
        attention_sum1 = self.uf(attention_sum_trans)

        # et (dense_input,bb)
        et = torch.tanh(hidden1 + encoder_outputs1 + attention_sum1)
        et = self.v(et)
        et = et.view(dense_input, -1)

        # et_div is attention alpha
        et_exp = torch.exp(et)
        et_sum = torch.sum(et_exp)
        et_div = et_exp/et_sum
        et_div = et_div.unsqueeze(0)

        # ct is context vector (1,128)
        ct = et_div*encoder_outputs
        ct = ct.sum(dim=1)
        ct = ct.sum(dim=1)
        ct = ct.unsqueeze(0)

        # the next hidden after gru
        hidden_next = self.gru(ct,st[0])
        hidden_next = hidden_next.unsqueeze(0)

        # compute the output (1,128)
        hidden2 = self.hidden2(hidden_next[0])
        embedded2 = self.emb2(embedded[0])
        ct2 = self.wc(ct)

        #output
        output = F.log_softmax(self.out(hidden2+embedded2+ct2), dim=1)
        output = output.unsqueeze(0)

        return output, hidden_next, et_div, attention_sum

    def initHidden(self):
        result = Variable(torch.randn(1, 1, self.hidden_size))
        return result.cuda()
