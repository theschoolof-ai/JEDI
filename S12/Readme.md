Exaplanation of the json file

The JSON file is basically a Key: Value pair dumped in a text format
The root key is the image_name and the value is attributes like 


**filename**: the name of the file with extension

**size**: space that the image takes

**regions**: this is an array consisting of the bounding boxes
shape_attributes: how the bbox is defined, could be a circle, rectangle, etc, its x, y coordinates and the height and width of the box

**region_attributes**: this contained the label for the region, here 'boots', and some meta data like if the image is blurry, has good illumination, etc

**file_attributes**: extra meta data of the image file, like the URL, caption and if it was from public domain

