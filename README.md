# Handwritten-Mathematical-Expression-Recognition (Pytorch)


**2019/8/13 README.md has been sorted out and you can see the previous version in version_before.md.**  
  
This program uses Attention and Coverage to realize **HMER** (HandWritten Mathematical Expression Recognition) and written by **Hongyu Wang** refer to Dr. Jianshu Zhang. Any discussion and questions are welcome to contact me (why0706@buaa.edu.cn).


# Requirements

	Python 3.6
	Pytorch == 1.0 

# Training and Testing
1. Install Requirements and pretrained Densenet weights can be download [here](https://download.pytorch.org/models/densenet121-a639ec97.pth))  .
2. Decompression files in **off\_image\_train** and **off\_image\_test**, and this will be your training data and testing data. 
3. python **'gen_pkl.py'**. This python file will compress your training pictures or testing pictures into a **'.pkl'** file. Moreover, you should write the correct location of your data files. 
4. python **'Train.py'** for training.
5. python **'Densenet_testway.py'** for testing.  
6. Open the source code of **HMER V2.0**. You can see detials in HMER_v2.0. 

# Experiment
+ This model is testing in CROHME 2016 dataset. All of my experiments are running in two TITAN XP GPUs. The batch_size is 6, the max len is 48 and the max Image size is 100000.  

+ The best result of this model is: 

+ > WER loss: **17.160%**  
 ExpRate: **38.595%**  

+ The HMER V2.0
![avatar](http://m.qpic.cn/psb?/V13MmUWH1KBoey/j8PBopZLNdZnNvCyyZRSPK.RWzFieO420uMgrUjEHtI!/b/dFIBAAAAAAAA&bo=iQNCAokDQgICOR0!&rf=viewer_4)
+ Visualization of results  

![avatar](http://r.photo.store.qq.com/psb?/V13MmUWH1KBoey/DpjTkIdquQo7zYbletKcv*EEPXZWipzxQuSiU53cw24!/r/dEcBAAAAAAAA)
![avatar](http://r.photo.store.qq.com/psb?/V13MmUWH1KBoey/Se*yEixUuODmf.g9J9ViJm85cWk7QwM6jEVij87cUxc!/r/dL4AAAAAAAAA)

+ Visualization of Attention

**Input image**  
![avatar](http://r.photo.store.qq.com/psb?/V13MmUWH1KBoey/1gc6vVDYrdNOwnYHhft3kMm0UjBQV8*sVxzaoOUixqY!/r/dL8AAAAAAAAA) 

**step by step**  
![avatar](http://m.qpic.cn/psb?/V13MmUWH1KBoey/Fv78zebr.kLV.TcsPurlB.LIhDE1t2GnDHcFm3vmYus!/b/dL8AAAAAAAAA&bo=TQIZAQAAAAADF2U!&rf=viewer_4)
![avatar](http://m.qpic.cn/psb?/V13MmUWH1KBoey/SDhLdfFBYFbZMsxUTTYFmuNHC6LxihjADY0QMog54.k!/b/dFQBAAAAAAAA&bo=TwIhAQAAAAADF18!&rf=viewer_4)
![avatar](http://m.qpic.cn/psb?/V13MmUWH1KBoey/plPANWRYY*0c3hAccSGMgtefee1hRMTUa.h*sYFoXEI!/b/dFIBAAAAAAAA&bo=UwIfAQAAAAADF30!&rf=viewer_4)
