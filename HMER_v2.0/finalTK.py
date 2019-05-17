'''
Python 3.6 
Pytorch >= 0.4
Written by Hongyu Wang and Wenliang Liu in Beihang university
'''
from tkinter import *
from tkinter.filedialog import askopenfilename, asksaveasfilename
from PIL import Image,ImageTk
import tkinter
from tkinter import messagebox
import numpy as np
import torch
from for_test_V20 import for_test
import matplotlib.pyplot as plt
import tkinter.font as tkfont

def imresize(im,sz):
	pil_im = Image.fromarray(im)
	return np.array(pil_im.resize(sz))

def resize( w_box, h_box, pil_image): 
	w, h = pil_image.size 
	f1 = 1.0*w_box/w 
	f2 = 1.0*h_box/h    
	factor = min([f1, f2])   
	width = int(w*factor)    
	height = int(h*factor)    
	return pil_image.resize((width, height), Image.ANTIALIAS)  
 
def choosepic():
	global Flag
	path_=askopenfilename()
	path.set(path_)
	global img_open
	img_open = Image.open(e1.get()).convert('L')
	img=ImageTk.PhotoImage(img_open)
	l1.config(image=img)
	l1.image=img #keep a reference
	var = tkinter.StringVar()
	var.set('                                                                               ')
	e2=Label(root,textvariable = var, font = ('Arial', 25))
	e2.place(relx=0.05,y=500)
	e3=Label(root,textvariable = var, font = ('Arial', 25))
	e3.place(relx=0.55,y=500)
	Flag = False


def trans():
	global img_open
	global prediction, attention
	var = tkinter.StringVar()
	var.set('                                                                               ')
	e3=Label(root,textvariable = var, font = ('Arial', 25))
	e3.place(relx=0.55,y=500)
	if Flag:
		print (messagebox.showerror(title='Error', message='No Image'))

	else:
		img_open2 = torch.from_numpy(np.array(img_open)).type(torch.FloatTensor)
		img_open2 = img_open2/255.0
		img_open2 = img_open2.unsqueeze(0)
		img_open2 = img_open2.unsqueeze(0)
		attention, prediction = for_test(img_open2)
		global prediction_string
		prediction_string = ''
		print(prediction_string)
		var = tkinter.StringVar()
		img_open = np.array(img_open)

		for i in range(attention.shape[0]):
			print(i)
			if prediction[i] == '<eol>':
				continue
			else:
				prediction_string = prediction_string + prediction[i]
			print(prediction_string)
			var.set(prediction_string)
			e3=Label(root,textvariable = var, font = ('Arial', 25))
			e3.place(relx=0.55,y=500)
			attention2 = imresize(attention[i,0,:,:],(img_open.shape[1],img_open.shape[0]))
			attention2 = attention2*attention2
			image_attention = img_open + attention2 * 1000
			image_open2 = Image.fromarray(image_attention)
			image_file = ImageTk.PhotoImage(image_open2)
			l2.config(image=image_file)
			l2.image=image_file #keep a reference
			l2.update()
			l2.after(500)
def trans1():
	global img_open
	global prediction, attention
	if Flag:

		print (messagebox.showerror(title='Error', message='No Image'))
	else:

		img_open2 = torch.from_numpy(np.array(img_open)).type(torch.FloatTensor)
		img_open2 = img_open2/255.0
		img_open2 = img_open2.unsqueeze(0)
		img_open2 = img_open2.unsqueeze(0)
		var = tkinter.StringVar()
		var.set('Detecting...')
		e2=Label(root,textvariable = var, font = ('Arial', 25))
		e2.place(relx=0.05,y=500)
		e2.update()
		attention, prediction = for_test(img_open2)
		global prediction_string
		prediction_string = ''
		print(prediction_string)
		
		img_open = np.array(img_open)

		for i in range(attention.shape[0]):
			if prediction[i] == '<eol>':
				continue
			else:
				prediction_string = prediction_string + prediction[i]
		print(prediction_string)
		var.set(prediction_string)
		e2=Label(root,textvariable = var, font = ('Arial', 25))
		e2.place(relx=0.05,y=500)
		image_file = ImageTk.PhotoImage(img_open)
		l1.config(image=image_file)
		l1.image=image_file 
		l1.update()

def trans2():
	global prediction_string
	if Flag:
		print (messagebox.showerror(title='Error', message='No Image'))
	else:
		fig = plt.figure(figsize=(3,3))
		ax = fig.add_subplot(111)
		ax.set_xticks([])
		ax.set_yticks([])
		ax.spines['left'].set_color('none')
		ax.spines['right'].set_color('none')
		ax.spines['bottom'].set_color('none')
		ax.spines['top'].set_color('none')
		prediction_image = '$' + prediction_string + '$'
		ax.text(0.1, 0.95, prediction_image, fontsize = 28)
		plt.show()


def saveClick():
	path=asksaveasfilename(filetypes = [('*.txt', '.txt')])
	with open (path,'w+') as fb:
		fb.write(prediction_string)


root=Tk()		
root.geometry('1600x900')
root.title('HMER Tool V2.0')
Flag=True


menubar = tkinter.Menu(root)
filemenu = tkinter.Menu(menubar, tearoff=0)
menubar.add_cascade(label='Start', font = ("Times New Roman", 15, 'bold'), menu=filemenu)
filemenu.add_command(label='Load',font = ("Times New Roman", 15, 'bold'), command=choosepic)
filemenu.add_command(label='Save',font = ("Times New Roman", 15, 'bold'), command=saveClick)
filemenu.add_separator()
filemenu.add_command(label='Exit', font = ("Times New Roman", 15, 'bold'), command=root.quit)


path=StringVar()
bu1 = Button(root,text='Start Detection', font = ('Times New Roman', 15, 'bold'), width = 18, height = 2, command=trans1)
bu2 = Button(root,text='Detail Results',font = ('Times New Roman', 15, 'bold'), width = 18, height = 2, command=trans)
bu3 = Button(root,text='Show Formula',font = ('Times New Roman', 15, 'bold'), width = 18, height = 2, command=trans2)
e1=Entry(root,state='readonly',text=path)
title = tkinter.Label(root, 
    text='HMER Tool V2.0',   
    font=('Times New Roman',40, 'bold'),    
    width=25, height=2 
    )
title_1 = tkinter.Label(root, 
    text='Your image:',   
    font=('Times New Roman', 25),    
    )
title_2 = tkinter.Label(root, 
    text='Result:',   
    font=('Times New Roman', 25),   
    )
title_3 = tkinter.Label(root, 
    text='Attention:',    
    font=('Times New Roman', 25),    
    )
title_4 = tkinter.Label(root, 
    text='Dynamic Result:',   
    font=('Times New Roman', 25),    
    )

title.place(relx = 0.5, y = 40, anchor = CENTER)    
bu1.place(relx=0.3,y=140, anchor = CENTER)
bu2.place(relx=0.5,y=140, anchor = CENTER)
bu3.place(relx=0.7,y=140, anchor = CENTER)

title_1.place(relx=0.05,y=200)
title_2.place(relx=0.05,y=450)
title_3.place(relx=0.55,y=200)
title_4.place(relx=0.55,y=450)


l1=Label(root)
l1.place(relx=0.05,y=250)
l2=Label(root)
l2.place(relx=0.55,y=250)
img_trans_show=Label(root)
img_trans_show.place(x=550,y=150)
root.config(menu=menubar)

root.mainloop()


