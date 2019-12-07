import os
import argparse
import xml.etree.ElementTree as ET

def convert_voc_annotation(data_path, data_type, anno_path, use_difficult_bbox=True):
    classes=['backboard','basket','basketball','person']  #classes id is decided by this order.
    # classes = ['bicycle', 'bus', 'car', 'motorbike']
#     classes = ['aeroplane','bicycle','bird','boat', 'bottle','bus','car','cat','chair','cow','diningtable','dog',
    #     horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor' ]

    img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', data_type + '.txt')
    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.strip() for line in txt]

    with open(anno_path, 'a') as f:
        for image_ind in image_inds:
            image_path = os.path.join(data_path, 'JPEGImages', image_ind + '.jpg') #从VOC数据中提起图片的信息
            annotation = image_path
            label_path = os.path.join(data_path, 'Annotations', image_ind + '.xml') #从VOC数据中提取注解信息
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            for obj in objects:
                difficult = obj.find('difficult').text.strip()
                if (not use_difficult_bbox) and(int(difficult) == 1):
                    continue
                bbox = obj.find('bndbox')
                class_ind = classes.index(obj.find('name').text.lower().strip())
                xmin = bbox.find('xmin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymin = bbox.find('ymin').text.strip()
                ymax = bbox.find('ymax').text.strip()                 #提取图片中目标的坐上右下信息
                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
            print(annotation)
            f.write(annotation + "\n")          #提取的信息写入默认注解路径
    return len(image_inds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default='/raid/my_data/')        #data path
    parser.add_argument("--train_annotation", default="./my_train.txt")        #save name
    parser.add_argument("--test_annotation",  default="./my_val.txt")           #save name
    flags = parser.parse_args()

    if os.path.exists(flags.train_annotation):os.remove(flags.train_annotation)
    if os.path.exists(flags.test_annotation):os.remove(flags.test_annotation)

    num1 = convert_voc_annotation(flags.data_path, 'train', flags.train_annotation, False)
    #num2 = convert_voc_annotation(os.path.join(flags.data_path, 'train/VOCdevkit/VOC2007'), 'trainval', flags.train_annotation, False)
    num3 = convert_voc_annotation(flags.data_path,  'val', flags.test_annotation, False)
    print('=> The number of image for train is: %d\tThe number of image for test is:%d' %(num1 , num3))
