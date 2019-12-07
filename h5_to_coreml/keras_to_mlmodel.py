import coremltools
import tensorflow as tf
import tfcoreml
# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
from keras.models import load_model
model = load_model('/home/sal/keras-yolo3-master/test/demo.h5')
print(model)



ml_model = coremltools.converters.keras.convert(model,input_names='input1',image_input_names='input1',
                                                input_name_shape_dict={'input1':[None,608,608,3]},
                                                output_names=['output1','output2','output3'],
                                                 image_scale=1/255.

                                                )
#zhu yi, input_name_shape_dice zhong:[None,H,W,C],shiyong h5 zhuan bu yong zhi dao wang luo jiegou,zhi xu yao zhi ding ren yi output_names.
ml_model.input_description['input1']='input image'
ml_model.output_description['output1']='The 13X13 grid(Scale)'
ml_model.output_description['output2']='The 26X26 grid(Scale)'
ml_model.output_description['output3']='The 52X52 grid(Scale)'
ml_model.save('yolov3_68.mlmodel')