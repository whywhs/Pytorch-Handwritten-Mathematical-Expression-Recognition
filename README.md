# Pytorch-Handwritten-Mathematical-Expression-Recognition

This program uses Attention and Coverage to realize **HMER** (HandWritten Mathematical Expression Recognition) and written by **Hongyu Wang**.

**Notice:**  
This program is writting with reference to the work of **Dr. Jianshu Zhang** from USTC. 

	@article{zhang2017watch,
	  title={Watch, attend and parse: An end-to-end neural network based approach to handwritten mathematical expression recognition},
	  author={Zhang, Jianshu and Du, Jun and Zhang, Shiliang and Liu, Dan and Hu, Yulong and Hu, Jinshui and Wei, Si and Dai, Lirong},
	  journal={Pattern Recognition},
	  volume={71},
	  pages={196--206},
	  year={2017},
	  publisher={Elsevier}
	}
	
	@article{zhang2018multi,
	  title={Multi-Scale Attention with Dense Encoder for Handwritten Mathematical Expression Recognition},
	  author={Zhang, Jianshu and Du, Jun and Dai, Lirong},
	  journal={arXiv preprint arXiv:1801.03530},
	  year={2018}
	}

# Requirements

	Python 3.6
	Pytorch 0.3 (This is important!)

# Training and Testing
1. Install Requirements.
2. Decompression files in **off\_image\_train** and **off\_image\_test**, and this will be your training data and testing data. 
3. python **'gen_pkl.py'**. This python file will compress your training pictures or testing pictures into a **'.pkl'** file. Moreover, you should write the correct location of your data files. 
4. python **'Train.py'** for training.
5. python **'Densenet_testway.py'** for testing.

# Experiment
+ This model is testing in CROHME14 dataset.

+ The best result of this model is: 

+ > WER loss: **25.715%**  
 ExpRate: **28.216%**  

+ The HMER V2.0
![avatar](http://m.qpic.cn/psb?/V13MmUWH1KBoey/j8PBopZLNdZnNvCyyZRSPK.RWzFieO420uMgrUjEHtI!/b/dFIBAAAAAAAA&bo=iQNCAokDQgICOR0!&rf=viewer_4)
+ Visualization of results  

![avatar](http://r.photo.store.qq.com/psb?/V13MmUWH1KBoey/DpjTkIdquQo7zYbletKcv*EEPXZWipzxQuSiU53cw24!/r/dEcBAAAAAAAA)
![avatar](http://r.photo.store.qq.com/psb?/V13MmUWH1KBoey/Se*yEixUuODmf.g9J9ViJm85cWk7QwM6jEVij87cUxc!/r/dL4AAAAAAAAA)

+ Visualization of Attention

**Input image**  
<div align=center>![avatar](http://r.photo.store.qq.com/psb?/V13MmUWH1KBoey/1gc6vVDYrdNOwnYHhft3kMm0UjBQV8*sVxzaoOUixqY!/r/dL8AAAAAAAAA)  
**step by step**  
<div align=center>![avatar](http://m.qpic.cn/psb?/V13MmUWH1KBoey/Fv78zebr.kLV.TcsPurlB.LIhDE1t2GnDHcFm3vmYus!/b/dL8AAAAAAAAA&bo=TQIZAQAAAAADF2U!&rf=viewer_4)
<div align=center>![avatar](http://m.qpic.cn/psb?/V13MmUWH1KBoey/SDhLdfFBYFbZMsxUTTYFmuNHC6LxihjADY0QMog54.k!/b/dFQBAAAAAAAA&bo=TwIhAQAAAAADF18!&rf=viewer_4)
<div align=center>![avatar](http://m.qpic.cn/psb?/V13MmUWH1KBoey/plPANWRYY*0c3hAccSGMgtefee1hRMTUa.h*sYFoXEI!/b/dFIBAAAAAAAA&bo=UwIfAQAAAAADF30!&rf=viewer_4)
