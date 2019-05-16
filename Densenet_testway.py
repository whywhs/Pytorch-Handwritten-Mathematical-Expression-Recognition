'''
Python 3.6 
Pytorch >= 0.4
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
from Attention_RNN import AttnDecoderRNN
from Densenet_torchvision import densenet121
from PIL import Image
from numpy import *

torch.backends.cudnn.benchmark = False

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
    return dist, len(label),hit_score,ins_score,del_score


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

valid_datasets=['./offline-test.pkl', './test_caption.txt']
dictionaries=['./dictionary.txt']
batch_Imagesize=16
valid_batch_Imagesize=16
batch_size_t=1
maxlen=48
maxImagesize=100000
hidden_size = 256
gpu = [0]

worddicts = load_dict(dictionaries[0])
worddicts_r = [None] * len(worddicts)
for kk, vv in worddicts.items():
        worddicts_r[vv] = kk

test,test_label = dataIterator(valid_datasets[0],valid_datasets[1],worddicts,batch_size=1,batch_Imagesize=batch_Imagesize,maxlen=maxlen,maxImagesize=maxImagesize)

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

off_image_test = custom_dset(test,test_label)
#print(off_image_train[10])

def imresize(im,sz):
    pil_im = Image.fromarray(im)
    return array(pil_im.resize(sz))

def collate_fn(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    img, label = zip(*batch)
    aa1 = 0
    bb1 = 0
    k = 0
    k1 = 0
    max_len = len(label[0])+1
    for j in range(len(img)):
        size = img[j].size()
        if size[1] > aa1:
            aa1 = size[1]
        if size[2] > bb1:
            bb1 = size[2]

    for ii in img:
        ii = ii.float()
        img_size_h = ii.size()[1]
        img_size_w = ii.size()[2]
        img_mask_sub_s = torch.ones(1,img_size_h,img_size_w).type(torch.FloatTensor)
        img_mask_sub_s = img_mask_sub_s*255.0
        img_mask_sub = torch.cat((ii,img_mask_sub_s),dim=0)
        padding_h = aa1-img_size_h
        padding_w = bb1-img_size_w
        m = torch.nn.ZeroPad2d((0,padding_w,0,padding_h))
        img_mask_sub_padding = m(img_mask_sub)
        img_mask_sub_padding = img_mask_sub_padding.unsqueeze(0)
        if k==0:
            img_padding_mask = img_mask_sub_padding
        else:
            img_padding_mask = torch.cat((img_padding_mask,img_mask_sub_padding),dim=0)
        k = k+1

    for ii1 in label:
        ii1 = ii1.long()
        ii1 = ii1.unsqueeze(0)
        ii1_len = ii1.size()[1]
        m = torch.nn.ZeroPad2d((0,max_len-ii1_len,0,0))
        ii1_padding = m(ii1)
        if k1 == 0:
            label_padding = ii1_padding
        else:
            label_padding = torch.cat((label_padding,ii1_padding),dim=0)
        k1 = k1+1

    img_padding_mask = img_padding_mask/255.0
    return img_padding_mask, label_padding

test_loader = torch.utils.data.DataLoader(
    dataset = off_image_test,
    batch_size = batch_size_t,
    shuffle = True,
    collate_fn = collate_fn
)

encoder = densenet121()
attn_decoder1 = AttnDecoderRNN(hidden_size,112,dropout_p=0.5)

encoder = torch.nn.DataParallel(encoder, device_ids=gpu)
attn_decoder1 = torch.nn.DataParallel(attn_decoder1, device_ids=gpu)
encoder = encoder.cuda()
attn_decoder1 = attn_decoder1.cuda()

encoder.load_state_dict(torch.load('model/encoder_lr0.00001_BN_te1_d05_SGD_bs8_mask_conv_bn_b.pkl'))
attn_decoder1.load_state_dict(torch.load('model/attn_decoder_lr0.00001_BN_te1_d05_SGD_bs8_mask_conv_bn_b.pkl'))

total_dist = 0
total_label = 0
total_line = 0
total_line_rec = 0
hit_all =0
ins_all =0
dls_all =0
wer_1 = 0
wer_2 = 0
wer_3 = 0
wer_4 = 0
wer_5 = 0
wer_6 = 0
wer_up=0

encoder.eval()
attn_decoder1.eval()


for step_t, (x_t, y_t) in enumerate(test_loader):
    x_real_high = x_t.size()[2]
    x_real_width = x_t.size()[3]
    if x_t.size()[0]<batch_size_t:
        break

    h_mask_t = []
    w_mask_t = []
    for i in x_t:
        #h*w
        size_mask_t = i[1].size()
        s_w_t = str(i[1][0])
        s_h_t = str(i[1][:,1])
        w_t = s_w_t.count('1')
        h_t = s_h_t.count('1')
        h_comp_t = int(h_t/16)+1
        w_comp_t = int(w_t/16)+1
        h_mask_t.append(h_comp_t)
        w_mask_t.append(w_comp_t)

    x_t = x_t.cuda()
    y_t = y_t.cuda()
    output_highfeature_t = encoder(x_t)

    x_mean_t = torch.mean(output_highfeature_t)
    x_mean_t = float(x_mean_t)
    output_area_t1 = output_highfeature_t.size()
    output_area_t = output_area_t1[3]
    dense_input = output_area_t1[2]

    decoder_input_t = torch.LongTensor([111]*batch_size_t)
    decoder_input_t = decoder_input_t.cuda()

    decoder_hidden_t = torch.randn(batch_size_t, 1, hidden_size).cuda()
    decoder_hidden_t = decoder_hidden_t * x_mean_t
    decoder_hidden_t = torch.tanh(decoder_hidden_t)

    prediction = torch.zeros(batch_size_t,maxlen)
    #label = torch.zeros(batch_size_t,maxlen)
    prediction_sub = []
    label_sub = []
    label_real = []
    prediction_real = []

    decoder_attention_t = torch.zeros(batch_size_t,1,dense_input,output_area_t).cuda()
    attention_sum_t = torch.zeros(batch_size_t,1,dense_input,output_area_t).cuda()

    m = torch.nn.ZeroPad2d((0,maxlen-y_t.size()[1],0,0))
    y_t = m(y_t)
    for i in range(maxlen):
        decoder_output, decoder_hidden_t, decoder_attention_t, attention_sum_t = attn_decoder1(decoder_input_t,
                                                                                         decoder_hidden_t,
                                                                                         output_highfeature_t,
                                                                                         output_area_t,
                                                                                         attention_sum_t,
                                                                                         decoder_attention_t,dense_input,batch_size_t,h_mask_t,w_mask_t,gpu)


        topv,topi = torch.max(decoder_output,2)
        if torch.sum(topi)==0:
            break
        decoder_input_t = topi
        decoder_input_t = decoder_input_t.view(batch_size_t)
        #print(topi.size()) 16,1

        # prediction
        prediction[:,i] = decoder_input_t


    for i in range(batch_size_t):
        for j in range(maxlen):
            if int(prediction[i][j]) ==0:
                break
            else:
                prediction_sub.append(int(prediction[i][j]))
                prediction_real.append(worddicts_r[int(prediction[i][j])])
        if len(prediction_sub)<maxlen:
            prediction_sub.append(0)

        for k in range(y_t.size()[1]):
            if int(y_t[i][k]) ==0:
                break
            else:
                label_sub.append(int(y_t[i][k]))
                label_real.append(worddicts_r[int(y_t[i][k])])
        label_sub.append(0)

        dist, llen, hit, ins, dls = cmp_result(label_sub, prediction_sub)
        wer_step = float(dist) / llen

        total_dist += dist
        total_label += llen
        total_line += 1
        if dist == 0:
            total_line_rec = total_line_rec+ 1

        print('step is %d' % (step_t))
        print('prediction is ')
        #print(''.join(prediction_real))
        print(prediction_real)
        print('the truth is')
        #print(''.join(label_real))
        print(label_real)
        print('the wer is %.5f' % (wer_step))

        label_sub = []
        prediction_sub = []
        label_real = []
        prediction_real = []


  
    # dist, llen, hit, ins, dls = cmp_result(label, prediction)
    # wer_step = float(dist) / llen
    # print('the wer is %.5f' % (wer_step))


    # if wer_step <= 0.1:
    #     wer_1 += 1
    # elif 0.1 < wer_step <= 0.2:
    #     wer_2 += 1
    # elif 0.2 < wer_step <= 0.3:
    #     wer_3 += 1
    # elif 0.3 < wer_step <= 0.4:
    #     wer_4 += 1
    # elif 0.4 < wer_step <= 0.5:
    #     wer_5 += 1
    # elif 0.5 < wer_step <= 0.6:
    #     wer_6 += 1
    # else:
    #     wer_up += 1

    # hit_all += hit
    # ins_all += ins
    # dls_all += dls
    # total_dist += dist
    # total_label += llen
    # total_line += 1
    # if dist == 0:
    #     total_line_rec += 1

wer = float(total_dist) / total_label
sacc = float(total_line_rec) / total_line
print('wer is %.5f' % (wer))
print('sacc is %.5f ' % (sacc))
# print('hit is %d' % (hit_all))
# print('ins is %d' % (ins_all))
# print('dls is %d' % (dls_all))
# print('wer loss is %.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f' % (wer_1, wer_2, wer_3, wer_4, wer_5, wer_6, wer_up))

