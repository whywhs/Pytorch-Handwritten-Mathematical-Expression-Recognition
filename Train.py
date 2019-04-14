'''
Python 3.6 
Pytorch 0.4
Written by Hongyu Wang in Beihang university
'''
import torch
import math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy
import torch.utils.data as data
from data_iterator import dataIterator
from Densenet_torchvision import densenet121
from Attention_RNN import AttnDecoderRNN
import random

# compute the wer loss
def cmp_result(label,rec):
    dist_mat = numpy.zeros((len(label)+1, len(rec)+1),dtype='int32')
    dist_mat[0,:] = range(len(rec) + 1)
    dist_mat[:,0] = range(len(label) + 1)
    for i in range(1, len(label) + 1):
        for j in range(1, len(rec) + 1):
            hit_score = dist_mat[i-1, j-1] + (label[i-1] != rec[j-1])
            ins_score = dist_mat[i,j-1] + 1
            del_score = dist_mat[i-1, j] + 1
            dist_mat[i,j] = min(hit_score, ins_score, del_score)
    dist = dist_mat[len(label), len(rec)]
    return dist, len(label)

def load_dict(dictFile):
    fp=open(dictFile)
    stuff=fp.readlines()
    fp.close()
    lexicon={}
    for l in stuff:
        w=l.strip().split()
        lexicon[w[0]]=int(w[1])
    print('total words/phones',len(lexicon))
    return lexicon

datasets=['./offline-train.pkl','./train_caption.txt']
valid_datasets=['./offline-test.pkl', './test_caption.txt']
dictionaries=['./dictionary.txt']
batch_Imagesize=500000
valid_batch_Imagesize=500000
batch_size=1
maxlen=48
maxImagesize= 100000
hidden_size = 256
teacher_forcing_ratio = 0.5

worddicts = load_dict(dictionaries[0])
worddicts_r = [None] * len(worddicts)
for kk, vv in worddicts.items():
        worddicts_r[vv] = kk

#load train data and test data
train,train_label = dataIterator(
                                    datasets[0], datasets[1],worddicts,batch_size=batch_size,
                                    batch_Imagesize=batch_Imagesize,maxlen=maxlen,maxImagesize=maxImagesize
                                 )
len_train = len(train)

test,test_label = dataIterator(
                                    valid_datasets[0],valid_datasets[1],worddicts,batch_size=batch_size,
                                    batch_Imagesize=batch_Imagesize,maxlen=maxlen,maxImagesize=maxImagesize
                                )


class custom_dset(data.Dataset):
    def __init__(self,train,train_label):
        self.train = train
        self.train_label = train_label

    def __getitem__(self, index):
        train_setting = torch.from_numpy(numpy.array(self.train[index]))
        label_setting = torch.from_numpy(numpy.array(self.train_label[index])).type(torch.LongTensor)

        size = train_setting.size()
        train_setting = train_setting.view(1,size[2],size[3])
        label_setting = label_setting.view(-1)
        return train_setting,label_setting

    def __len__(self):
        return len(self.train)

off_image_train = custom_dset(train,train_label)
off_image_test = custom_dset(test,test_label)

# collate_fn is writting for padding imgs in batch. But now, I used batch_size=1, so this function has no effect. 
def collate_fn(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    img, label = zip(*batch)
    aa1 = 0
    bb1 = 0
    max_len = len(label[0])
    for j in range(len(img)):
        size = img[j].size()
        if size[1] > aa1:
            aa1 = size[1]
        if size[2] > bb1:
            bb1 = size[2]
    img_padding = torch.zeros(len(img),1,aa1,bb1).type(torch.FloatTensor)
    img_mask = torch.zeros(len(img),1,aa1,bb1).type(torch.FloatTensor)
    for ii in range (len(img)):
        size = img[ii].size()
        for ii1 in range (size[1]):
            for ii2 in range (size[2]):
                img_padding[ii][0][ii1][ii2] = img[ii][0][ii1][ii2]
                img_mask[ii][0][ii1][ii2] = 1
    img_padding = img_padding/255
    # img_padding_mask = torch.cat((img_padding,img_mask),1)

    label_padding = torch.zeros(len(label),max_len+1).type(torch.LongTensor)
    for i in range(len(label)):
        for i1 in range(len(label[i])):
            label_padding[i][i1] = label[i][i1]

    return img_padding, label_padding

train_loader = torch.utils.data.DataLoader(
    dataset = off_image_train,
    batch_size = batch_size,
    shuffle = True,
    collate_fn = collate_fn,
    num_workers=8,
    )
test_loader = torch.utils.data.DataLoader(
    dataset = off_image_test,
    batch_size = batch_size,
    shuffle = True,
    collate_fn = collate_fn,
    num_workers=8,
)

def my_train(target_length,attn_decoder1,
             output_highfeature, output_area,y,criterion,encoder_optimizer1,decoder_optimizer1,x_mean,dense_input):
    loss = 0

    # teacher_forcing is very useful in training RNN.
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    decoder_input = Variable(torch.LongTensor([[111]]))
    decoder_input = decoder_input.cuda()
    decoder_hidden = attn_decoder1.initHidden()
    decoder_hidden = decoder_hidden*x_mean
    decoder_hidden = torch.tanh(decoder_hidden)
    attention_sum = Variable(torch.zeros(1,dense_input,output_area).cuda())
    decoder_attention = Variable(torch.zeros(1,dense_input,output_area).cuda())

    if use_teacher_forcing:
        encoder_optimizer1.zero_grad()
        decoder_optimizer1.zero_grad()
        my_num = 0

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention, attention_sum = attn_decoder1(decoder_input,
                                                                                             decoder_hidden,
                                                                                             output_highfeature,
                                                                                             output_area,
                                                                                             attention_sum,
                                                                                             decoder_attention,
                                                                                             dense_input)

            loss += criterion(decoder_output[0], y[:,di])
            my_num = my_num + 1
            if int(y[0][di]) == 0:
                break
            decoder_input = y[:,di]

        loss.backward()
        encoder_optimizer1.step()
        decoder_optimizer1.step()
        return loss.item()

    else:
        encoder_optimizer1.zero_grad()
        decoder_optimizer1.zero_grad()
        my_num = 0

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention,attention_sum= attn_decoder1(decoder_input, decoder_hidden,
                                                                                output_highfeature, output_area,
                                                                                attention_sum,decoder_attention,dense_input)
            #print(decoder_output.size()) 1*1*112
            #print(y.size())  1*37
            topv, topi = decoder_output[0][0].topk(1)
            decoder_input = topi
            loss += criterion(decoder_output[0], y[:,di])
            my_num = my_num + 1

            # if int(topi[0]) == 0:
            #     break

        loss.backward()
        encoder_optimizer1.step()
        decoder_optimizer1.step()
        return loss.item()

#encoder = DenseNet121().cuda()
encoder = densenet121().cuda()


pthfile = r'densenet121-a639ec97.pth'
pretrained_dict = torch.load(pthfile) 
encoder_dict = encoder.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
encoder_dict.update(pretrained_dict)
encoder.load_state_dict(encoder_dict)

attn_decoder1 = AttnDecoderRNN(hidden_size,112,dropout_p=0.2).cuda()
# attn_pre = torch.load('model/attn_decoder_lr0.00009_nopadding_baseline.pkl')
# attn_dict = attn_decoder1.state_dict()
# attn_pre = {k: v for k, v in attn_pre.items() if k in attn_dict}
# attn_dict.update(attn_pre)
# attn_decoder1.load_state_dict(attn_dict)
# encoder.load_state_dict(torch.load('model/encoder_lr0.00009_nopadding.pkl'))
# attn_decoder1.load_state_dict(torch.load('model/attn_decoder_lr0.00009_nopadding.pkl'))

lr_rate = 0.00009
encoder_optimizer1 = torch.optim.Adam(encoder.parameters(), lr=lr_rate)
decoder_optimizer1 = torch.optim.Adam(attn_decoder1.parameters(), lr=lr_rate)

criterion = nn.CrossEntropyLoss()
exprate = 0
#encoder.load_state_dict(torch.load('model/encoder_lr0.00009_nopadding_pre_GN_te05_d02.pkl'))
#attn_decoder1.load_state_dict(torch.load('model/attn_decoder_lr0.00009_nopadding_pre_GN_te05_d02.pkl'))

for epoch in range(1000):

    # if using SGD optimizer
    # if epoch%8 == 0:
    #     lr_rate = lr_rate/10
    # encoder_optimizer1 = torch.optim.SGD(encoder.parameters(), lr=lr_rate,momentum=0.9)
    # decoder_optimizer1 = torch.optim.SGD(attn_decoder1.parameters(), lr=lr_rate,momentum=0.9)

    running_loss=0
    whole_loss = 0

    encoder.train(mode=True)
    attn_decoder1.train(mode=True)

    # this is the train
    for step,(x,y) in enumerate(train_loader):
        x = Variable(x.cuda())
        y = Variable(y.cuda())

        # out is CNN featuremaps
        out = encoder(x)
        output_highfeature = out.squeeze(0)

        x_mean = torch.mean(output_highfeature)
        x_mean = float(x_mean)

        # dense_input is height and output_area is width which is bb
        output_area1 = output_highfeature.size()
        output_area = output_area1[2]
        dense_input = output_area1[1]
        target_length = y.size()[1]

        running_loss += my_train(target_length,attn_decoder1,output_highfeature,
                                output_area,y,criterion,encoder_optimizer1,decoder_optimizer1,x_mean,dense_input)

        
        if step % 100 == 99:
            pre = ((step+1)/len_train)*100
            whole_loss += running_loss
            running_loss = running_loss/100
            print('epoch is %d, loading for %.3f%%, running_loss is %f' %(epoch,pre,running_loss))
            with open("training_data/running_loss_%.5f_pre_GN_te05_d02.txt" %(lr_rate),"a") as f:
                f.write("%s\n"%(str(running_loss)))
            running_loss = 0

    loss_all_out = whole_loss / len_train
    print("epoch is %d, the whole loss is %f" % (epoch, loss_all_out))
    with open("training_data/whole_loss_%.5f_pre_GN_te05_d02.txt" % (lr_rate), "a") as f:
        f.write("%s\n" % (str(loss_all_out)))

    # this is the prediction and compute wer loss
    total_dist = 0
    total_label = 0
    total_line = 0
    total_line_rec = 0

    encoder.eval()
    attn_decoder1.eval()

    for step_t, (x_t, y_t) in enumerate(test_loader):
        x_t = Variable(x_t.cuda())
        y_t = Variable(y_t.cuda())
        out_t = encoder(x_t)
        output_highfeature_t = out_t.squeeze(0)
        x_mean_t = torch.mean(output_highfeature_t)
        x_mean_t = float(x_mean_t)
        output_area_t1 = output_highfeature_t.size()
        output_area_t = output_area_t1[2]
        dense_input = output_area_t1[1]
        target_length_t = y_t.size()[1]

        decoder_input_t = Variable(torch.LongTensor([[111]]))
        decoder_input_t = decoder_input_t.cuda()
        decoder_hidden_t = attn_decoder1.initHidden()
        decoder_hidden_t = decoder_hidden_t * x_mean_t
        decoder_hidden_t = torch.tanh(decoder_hidden_t)

        prediction = []
        label = []
        decoder_attention_t = Variable(torch.zeros(1,dense_input,output_area_t).cuda())
        attention_sum_t = Variable(torch.zeros(1,dense_input,output_area_t).cuda())
        for i in range(48):
            decoder_output, decoder_hidden_t, decoder_attention_t, attention_sum_t = attn_decoder1(decoder_input_t,
                                                                                             decoder_hidden_t,
                                                                                             output_highfeature_t,
                                                                                             output_area_t,
                                                                                             attention_sum_t,
                                                                                             decoder_attention_t,dense_input)
            topv, topi = decoder_output[0].topk(1)
            decoder_input_t = topi

            # prediction
            prediction.append(int(topi[0]))
            if int(topi[0]) == 0:
                break
        # label
        for i_label in range(target_length_t):
            label.append(int(y_t[0][i_label]))
        #label.append(0)

        dist, llen = cmp_result(label, prediction)
        total_dist += dist
        total_label += llen
        total_line += 1
        if dist == 0:
            total_line_rec = total_line_rec+ 1

    print('total_line_rec is',total_line_rec)
    wer = float(total_dist) / total_label
    sacc = float(total_line_rec) / total_line
    print('wer is %.5f' % (wer))
    print('sacc is %.5f ' % (sacc))
    with open("training_data/wer_%.5f_pre_GN_te05_d02.txt" % (lr_rate), "a") as f:
        f.write("%s\n" % (str(wer)))


    if (sacc > exprate):
        exprate = sacc
        print(exprate)
        print("saving the model....")
        print('encoder_lr%.5f_nopadding_pre_GN_te05_d02_f.pkl' %(lr_rate))
        torch.save(encoder.state_dict(), 'model/encoder_lr%.5f_nopadding_pre_GN_te05_d02_f.pkl'%(lr_rate))
        torch.save(attn_decoder1.state_dict(), 'model/attn_decoder_lr%.5f_nopadding_pre_GN_te05_d02_f.pkl'%(lr_rate))
        print("done")
    else:
        print('the best is %f' % (exprate))
        print('the loss is bigger than before,so do not save the model')










