# JEDI GROUP

**Team members:**
Canvas name| email id |
---|---|
Thamizhannal| annalwins@gmail.com|
Vikas Kumar| vikasmech.nitk@gmail.com|
Raga Ashritha| ragaashritha@gmail.com|
Pranava Sai| pranavbalasankula@gmail.com|


## Link to the Google drive with the final customdata: https://drive.google.com/drive/u/0/folders/1wPxFaaus2FQZ7aMoBif1eqtxW4mcCUfj
## customdata description: 

This the data of images regarding the PPE kit, primarily for construction workers. It has 4 classes which are **`hardhat, vest, mask, boots`**. Below is the definition of the all the files in the shared google drive

**Total number of Images:** `3489`

file name| description|
---|---|
images| Input images that are untouched|
labels| contains 1 .txt file per image with inforamtion of class cx(bounding box centre) cy(bounding box centre) h(ratio to reduce anchor box height) w(ratio to reduce anchor box width) |
train.txt, test.txt|text files with location of training and text images respectively|
train.shapes, test.shapes|text files with shapes of training and text images respectively|
custom.data|File describing the number of classes and local to train.txt and test.txt|
custom.names| text file with class label descriptions|
midas_out_colormap|Images which have depth inforamtion of every image from the images folder on a colourful scale|
midas_out_greyscale|Images which have depth inforamtion of every image from the images folder on a grey scale|
plane_rcnn_inference|Images which have plane segmentation inforamtion of every image from the images folder|

