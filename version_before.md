# Pytorch-Handwritten-Mathematical-Expression-Recognition

## Update in 2019/5/16 (An important update in my program!):
**2019/5/19 This code is running in Pytorch >= 1.0. If you use Pytorch0.4, something will be wrong when you load your saving model. Then, I updated some codes.**  

**2019/5/17 HMER V2.0 is open!**  

**First of all, I am very grateful to those who have used my program, and I am sorry that the previous code is not perfect. So, I updated my code to make it better for those who need it.**  
  
1、Now, my program is much faster and much more accurate than before. 
+ > Previous: **2h 30min for an epoch**  
Now: **less than 15min for an epoch**
+ > WER Loss: **17.160%**  
ExpRate: **38.595%**  

2、Add **batch_size** in my code, and **Supporting Multi-GPU Parallel Operations**.  
All of my experiments are running in two TITAN XP GPUs. The batch_size is 6, the max len is 48 and the max Image size is 100000. You can try larger if you have enough GPU memory.  

3、Open the source code of **HMER V2.0**. You can see detials in HMER_v2.0.  

4、Although the code is much better now than before, there are still many improvements. For example, you can try different optimization functions(Now is SGD) or different batch_size. If you have some good ideas and improve the results, you can contact to me. Meanwhile, I will also try some different and always pay attention to my code.  

## Update in 2019/3/27:  

1、Now, this program is running in Pytorch0.4.   
2、Use pretrained Densenet weights.(You can download [here](https://download.pytorch.org/models/densenet121-a639ec97.pth))  
3、Solve some BUGs. (sacc is always 0).  
4、Improve the accuracy of recognition. 
+ > WER loss: **24.097%**  
 ExpRate: **32.216%**  
  
5、Any discussion and questions are welcome to contact me (why0706@buaa.edu.cn).
  
## Original
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
	Pytorch >= 0.4 

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
![avatar](http://r.photo.store.qq.com/psb?/V13MmUWH1KBoey/1gc6vVDYrdNOwnYHhft3kMm0UjBQV8*sVxzaoOUixqY!/r/dL8AAAAAAAAA) 

**step by step**  
![avatar](http://m.qpic.cn/psb?/V13MmUWH1KBoey/Fv78zebr.kLV.TcsPurlB.LIhDE1t2GnDHcFm3vmYus!/b/dL8AAAAAAAAA&bo=TQIZAQAAAAADF2U!&rf=viewer_4)
![avatar](http://m.qpic.cn/psb?/V13MmUWH1KBoey/SDhLdfFBYFbZMsxUTTYFmuNHC6LxihjADY0QMog54.k!/b/dFQBAAAAAAAA&bo=TwIhAQAAAAADF18!&rf=viewer_4)
![avatar](http://m.qpic.cn/psb?/V13MmUWH1KBoey/plPANWRYY*0c3hAccSGMgtefee1hRMTUa.h*sYFoXEI!/b/dFIBAAAAAAAA&bo=UwIfAQAAAAADF30!&rf=viewer_4)
