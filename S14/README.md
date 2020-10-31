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
labels| contains 1 .txt file per image with information of class cx(bounding box centre) cy(bounding box centre) h(ratio to reduce anchor box height) w(ratio to reduce anchor box width) |
train.txt, test.txt|text files with location of training and text images respectively|
train.shapes, test.shapes|text files with shapes of training and text images respectively|
custom.data|File describing the number of classes and local to train.txt and test.txt|
custom.names| text file with class label descriptions|
midas_out_colormap|Images which have depth information of every image from the images folder on a colourful scale. Implemented from the repo: [MiDas](https://github.com/intel-isl/MiDaS) using the colab file: [assign_14a](https://github.com/theschoolof-ai/JEDI/blob/master/S14/Session14_MiDas.ipynb)|
midas_out_greyscale|Images which have depth information of every image from the images folder on a grey scale. Implemented from the repo: [MiDas](https://github.com/intel-isl/MiDaS) using the colab file: [assign_14a](https://github.com/theschoolof-ai/JEDI/blob/master/S14/Session14_MiDas.ipynb)|
plane_rcnn_inference|Images which have plane segmentation information of every image from the images folder. PlaneR-CNN, that detects arbitrary number of planes, and reconstructs piecewise planar surfaces from a single RGB image was implemented from the repo: [planercnn](https://github.com/NVlabs/planercnn) using the following colab file: [assign_14b](https://github.com/theschoolof-ai/JEDI/blob/master/S14/assignment_EVA5_JEDI_14b.ipynb)|

