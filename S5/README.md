# JEDI GROUP

**Team members:**
Canvas name| email id |
---|---|
Thamizhannal| annalwins@gmail.com|
Vikas Kumar| vikasmech.nitk@gmail.com|
Raga Ashritha| ragaashritha@gmail.com|
Pranava Sai| pranavbalasankula@gmail.com|

# Iteration1

### Target:
Basic architecture with good potential 

### Results:
1. Parameters: 15,918
2. Best Train Accuracy: 99.24%
3. Best Test Accuracy: 98.86%

### Analysis:
1. Model seems to show promise with highest Train and Test accuracy not differing much.
2. Slight Over-fitting in my opinion as Train accuracy is increasing with epochs but Test accuracy got stagnent after 6-7 epochs
3. Need to decrease the number of parameters as well.

### Filelink: https://github.com/theschoolof-ai/JEDI/blob/master/S5/EVA5%20-%20Session%205_1.ipynb 
--------

# Iteration2
### Target:
Decrease the size of parameters by replacing last big 7x7 convolution with global average pooling and acheive more efficiency and remove the slight overfit by adding regularization in the form of batch normalization.

### Results:
1. Parameters: 11,228
2. Best Train Accuracy: 99.37%
3. Best Test Accuracy: 99.19%

### Analysis:
1. There is need for more parameters when last layer conv layer was replaced by gap
2. Decrease parameter from top of the architecture(If it does not change the performace of the model a lot) and add layer post GAP. 
3. There is need for increase of capacity of model without increase in paramters - Augmentation needs to come to rescue here

### Filelink: https://github.com/theschoolof-ai/JEDI/blob/master/S5/EVA5%20-%20Session%205_2.ipynb
----------
# Iteration3 
### Target:
Decrease the parameters in the first 2 layers and add a 1x1 post GAP. Reach 99.4% accuracy with the power of image agumentation. Try to find the better Batch size for this Architecture in 32, 64, 128, 256.

### Results:
1. Parameters: 7,917
2. Best Train Accuracy: 99.24%
3. Best Test Accuracy: 99.48%

### Analysis:
1. Image augmentation works as good regularization
2. 64 batch size showcases better performance than 32, 64, 128, 256
3. Model takes more time to converge with LR - 0.01 and was able to reaach > 99.4% accuracy only once in 14th epoch.
4. Adding more regularisation can make the model robust and perform consistently on test data.

### Filelink: https://github.com/theschoolof-ai/JEDI/blob/master/S5/EVA5%20-%20Session%205_3.ipynb
-------------
# Iteration4 
### Target:
Reach 99.4% accuracy consistently in last 3-4 epochs by a applying more regularization in the form of dropout and increase LR rate to understand if convergence occurs faster and use LR schedular to change LR wherever Loss hit a plateau

### Results:
1. Parameters: 7,917
2. Best Train Accuracy: 99.08%
3. Best Test Accuracy: 99.43% 
4. greater than 99.4% in last 4 epochs(12-15)

### Analysis:
1. Dropout added the necessary regularization required to make model robust
2. Though increased LR was helping in fast convergence but after 5-6 epochs it would tapper off implying a need for smaller LR so used a step LR schedule at after 4 epochs and decreased it to 0.8 times the previous LR
3. All the changes added to the base CNN architechture to acheive the target accuracy 
    - Input image normalization
    - Image augmentation - rotation for Train data 
    - Batch Normalization
    - Dropouts 
    - LR scheduler 
    - GAP

### Filelink: https://github.com/theschoolof-ai/JEDI/blob/master/S5/EVA5%20-%20Session%205_4.ipynb



 
