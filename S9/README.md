# JEDI GROUP

**Team members:**
Canvas name| email id |
---|---|
Thamizhannal| annalwins@gmail.com|
Vikas Kumar| vikasmech.nitk@gmail.com|
Raga Ashritha| ragaashritha@gmail.com|
Pranava Sai| pranavbalasankula@gmail.com|


## Modular Implementation

Refer to [tsai.jedi](https://github.com/theschoolof-ai/JEDI/tree/master/tsai.jedi) understand the contents in different modules that have been used to run the s9.ipynb. Here is an overview

- **batchnorm.py:** GBN implementaion
- **dataloader.py:** - train and test data loader for MNIST, CIFAR10
- **Engine_train_test.py:** - train and test functions 
- **Models/:** Folder for architecture creation and view for every session
- **config.py:** parameters needs to set to run the following main file. Will be adding further to config when notebook would be created
- **model_objects/**: Folder for saving trained models
- **datatransforms:** Training and testing transformations. Albumentation transformation can be observed in this module
- **gradcam** - Implementation to generate and view gradcam of images
- **aftereffects** - Module with implementations like viewing misclassified or correctly classified images, Image plot generators e.t.c which are useful post training model



### Target:
Your Target is 87% accuracy. No limit on the number of epochs. Use default ResNet18 code (so params are fixed). 



### Results:

1. Parameters:  11,173,962 
2. Best Train Accuracy: 99.87%
3. Best Test Accuracy: 87.87%
4. greater than 87% validation accuracy post 21 epochs
