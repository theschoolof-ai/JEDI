# JEDI GROUP

**Team members:**
Canvas name| email id |
---|---|
Thamizhannal| annalwins@gmail.com|
Vikas Kumar| vikasmech.nitk@gmail.com|
Raga Ashritha| ragaashritha@gmail.com|
Pranava Sai| pranavbalasankula@gmail.com|


## Modular Implementation

We have created a eva5-jedi project and exported it as open source https://test.pypi.org/project/eva5-jedi/0.0.1/

Refer to [tsai.jedi](https://github.com/theschoolof-ai/JEDI/tree/master/tsai.jedi) if you need to refer to the contents of the package
It consist of following modules

- **batchnorm.py:** GBN implementation, depthwise convolution implementation
- **dataloader.py:** - train and test data loader
- **Engine_train_test.py:** - train and test functions
- **model.py:** Architecture creation and view
- **config.py:** parameters needed to run the main or colab notebook



### Target:

Reached 80% of validation accuracy  in 20th epoch and it is consistent in next consecutive epochs too.  We used GBN, Convolution, Depth wise separable convolution, Dilated convolution and Global average pooling in our network. 



### Results:

1. Parameters:  642,778 
2. Best Train Accuracy: 82.53%
3. Best Test Accuracy: 82.25%
4. greater than 80% validation accuracy from 21-30th epochs.

### Analysis:

1. Dropout added the necessary regularization required to make model robust
2. Also Image augumentation & GBN techniques helped to increase validation accuracy.
### All the changes added to the base CNN architechture to acheive the target accuracy

1. Input image normalization
2. Image augmentation - rotation for Train data
3. Batch Normalization
4. Dropouts
5. Convolition, Depthwise separable convolution, Dilated convolution
6. LR scheduler
7. GAP
