# JEDI GROUP

**Team members:**
Canvas name| email id |
---|---|
Thamizhannal| annalwins@gmail.com|
Vikas Kumar| vikasmech.nitk@gmail.com|
Raga Ashritha| ragaashritha@gmail.com|
Pranava Sai| pranavbalasankula@gmail.com|


## Modular Implementation

Refer to [tsai.jedi](https://github.com/theschoolof-ai/JEDI/tree/master/tsai.jedi) understand the contents in different modules that have been used to run the s8.ipynb. Here is an overview

- **batchnorm.py:** GBN implementaion
- **dataloader.py:** - train and test data loader 
- **Engine_train_test.py:** - train and test functions 
- **Models/:** Folder for architecture creation and view for every session
- **config.py:** parameters needs to set to run the following main file. Will be adding further to config when notebook would be created
- **model_objects/**: Folder for saving models



### Target:
Your Target is 85% accuracy. No limit on the number of epochs. Use default ResNet18 code (so params are fixed). 



### Results:

1. Parameters:  11,173,962 
2. Best Train Accuracy: 95.53%
3. Best Test Accuracy: 90.90%
4. greater than 85% validation accuracy from 12-30th epochs.


### All the changes added to the base resnet CNN architechture to acheive the target accuracy

1. Input image normalization
2. Image augmentation - rotation for Train data
3. Batch Normalization
4. LR scheduler




