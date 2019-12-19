reference to https://github.com/qqwweee/keras-yolo3

I simplify some details,add process_data file,
and you can use two .py produce suitable data format to train.

gengerate_anchors.py could obtain your data anchors.
just modifiy some parameters.

tips:
this file--coco_annotation.py could extract annotations from coco.json file,because coco.class_id is not continue,it has
80 classes,but the number id to 90.

more details refer to my blog --
