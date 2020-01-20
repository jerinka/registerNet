# reg1_net
Image registeration of moved image to a single fixed image usig CNN! 
CNN estimates affine transform(change parameters to do perspective also) to match input moved image to the template/fixed image used for training. 

#Run:
train.py
test.py


#Notebook:
Open terminal
Enter: jupyter notebook
Open: register_single_cnn.ipynb


#To Do
-Adding option of perspective
-Adding proper data generator to use parallel processing
-Adding moved image folder for training CNN with real moved images in addition to augmented ones.

![Image description](https://github.com/jerinka/reg1_net/blob/master/an1.jpg)
