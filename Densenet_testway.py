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
from Attention_RNN import AttnDecoderRNN
#from Resnet101 import resnet101
from Densenet_torchvision import densenet121
#from Densenet import DenseNet121
from PIL import Image
from numpy import *
#from pylab import *
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker

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
batch_size=1
maxlen=48
maxImagesize=100000
hidden_size = 256

worddicts = load_dict(dictionaries[0])
worddicts_r = [None] * len(worddicts)
for kk, vv in worddicts.items():
        worddicts_r[vv] = kk

test,test_label = dataIterator(valid_datasets[0],valid_datasets[1],worddicts,batch_size=batch_size,batch_Imagesize=batch_Imagesize,maxlen=maxlen,maxImagesize=maxImagesize)

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

    label_padding = torch.zeros(len(label),max_len).type(torch.LongTensor)
    for i in range(len(label)):
        for i1 in range(len(label[i])):
            label_padding[i][i1] = label[i][i1]

    return img_padding, label_padding

test_loader = torch.utils.data.DataLoader(
    dataset = off_image_test,
    batch_size = batch_size,
    shuffle = True,
    collate_fn = collate_fn
)

#encoder = DenseNet121().cuda()
#encoder = resnet101().cuda()
encoder = densenet121().cuda()
attn_decoder1 = AttnDecoderRNN(hidden_size,112,dropout_p=0.1).cuda()
encoder.load_state_dict(torch.load('model/encoder_lr0.00009_nopadding_pre_GN_te05_d02_f.pkl'))
attn_decoder1.load_state_dict(torch.load('model/attn_decoder_lr0.00009_nopadding_pre_GN_te05_d02_f.pkl'))

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
    cnn_out=[]
    cnn_out1=[]
    x_t = Variable(x_t.cuda())
    x_real_high = x_t.size()[2]
    x_real_width = x_t.size()[3]
    x_real = x_t.view(x_real_high,x_real_width)
    y_t = Variable(y_t.cuda())
    out_t = encoder(x_t)
    #out_1_conv = out_1_conv.squeeze(0)
    output_highfeature_t = out_t.squeeze(0)
    x_mean_t = torch.mean(output_highfeature_t)
    x_mean_t = float(x_mean_t)
    output_area_t1 = output_highfeature_t.size()  # x1 4*2944*-1
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
    decoder_attention_t_cat = []
    decoder_attention_t = Variable(torch.zeros(1, dense_input, output_area_t).cuda())
    attention_sum_t = Variable(torch.zeros(1, dense_input, output_area_t).cuda())
    for i in range(maxlen+1+1):
        decoder_output, decoder_hidden_t, decoder_attention_t, attention_sum_t = attn_decoder1(decoder_input_t,
                                                                                               decoder_hidden_t,
                                                                                               output_highfeature_t,
                                                                                               output_area_t,
                                                                                               attention_sum_t,
                                                                                               decoder_attention_t,
                                                                                               dense_input)
        decoder_attention_t_cat.append(decoder_attention_t.data.cpu().numpy())
        # decoder_attention_t_cat = decoder_attention_t_cat
        topv, topi = decoder_output[0].topk(1)
        decoder_input_t = topi
        prediction.append(int(topi[0]))
        if int(topi[0]) == 0:
            break
    for i_label in range(target_length_t):
        label.append(int(y_t[0][i_label]))
    label.append(0)

    k = numpy.array(decoder_attention_t_cat)
    x_real = numpy.array(x_real.cpu().data)

    prediction_real = []
    label_real = []

    for ir in range(len(prediction)):
        prediction_real.append(worddicts_r[prediction[ir]])
    for ir1 in range(len(label)):
        label_real.append(worddicts_r[label[ir1]])
    print('step is %d' % (step_t))
    print('prediction is ')
    #print(''.join(prediction_real))
    print(prediction_real)
    print('the truth is')
    #print(''.join(label_real))
    print(label_real)

    dist, llen, hit, ins, dls = cmp_result(label, prediction)
    wer_step = float(dist) / llen
    print('the wer is %.5f' % (wer_step))

    '''
    # RNN show
    for ki in range(k.shape[0]):
        k_show = k[ki][0]
        k_show = imresize(k_show,(x_real_width,x_real_high))
        k_max = k_show.max()
        k_len = 1/k_max
        k_show1 = k_show*k_len
        k_show1 = k_show1*k_show1
        # plt.imshow(k_show1, interpolation='nearest', cmap='gray_r')
        # plt.colorbar(shrink=.92)
        # plt.xticks(())
        # plt.yticks(())
        # plt.show()

        plt.imshow(k_show1+x_real, interpolation='nearest', vmin=0,vmax=1.5,cmap='gray_r')
        #plt.imshow(x_real, interpolation='nearest', cmap='gray_r')
        plt.colorbar(shrink=.92)
        plt.xticks(())
        plt.yticks(())
        plt.show()
    '''

    '''
    # CNN show
    cnn_out.append(out_1_conv.data.cpu().numpy())
    cnn_out = numpy.array(cnn_out)
    for cnni in range(48):
        cnn_show = cnn_out[0][cnni]
        plt.imshow(cnn_show, interpolation='nearest', cmap='gray_r')
        plt.colorbar(shrink=.92)
        plt.xticks(())
        plt.yticks(())
        plt.show()

    cnn_out1.append(output_highfeature_t.data.cpu().numpy())
    cnn_out1 = numpy.array(cnn_out1)
    for cnni in range(648):
        cnn_show1 = cnn_out1[0][cnni]
        plt.imshow(cnn_show1, interpolation='nearest', cmap='gray_r')
        plt.colorbar(shrink=.92)
        plt.xticks(())
        plt.yticks(())
        plt.show()
    '''

    if wer_step <= 0.1:
        wer_1 += 1
    elif 0.1 < wer_step <= 0.2:
        wer_2 += 1
    elif 0.2 < wer_step <= 0.3:
        wer_3 += 1
    elif 0.3 < wer_step <= 0.4:
        wer_4 += 1
    elif 0.4 < wer_step <= 0.5:
        wer_5 += 1
    elif 0.5 < wer_step <= 0.6:
        wer_6 += 1
    else:
        wer_up += 1

    hit_all += hit
    ins_all += ins
    dls_all += dls
    total_dist += dist
    total_label += llen
    total_line += 1
    if dist == 0:
        total_line_rec += 1

wer = float(total_dist) / total_label
sacc = float(total_line_rec) / total_line
print('wer is %.5f' % (wer))
print('sacc is %.5f ' % (sacc))
print('hit is %d' % (hit_all))
print('ins is %d' % (ins_all))
print('dls is %d' % (dls_all))
print('wer loss is %.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f' % (wer_1, wer_2, wer_3, wer_4, wer_5, wer_6, wer_up))

