import tensorflow as tf
import os
from core.yolov3 import YOLOV3
from from_darknet_weights_to_ckpt import load_weights
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

input_size = 608
darknet_weights = '/home/sal/h5_to_weight_yolo3-master/yolov3.weights'
pb_file = './yolov3_608.pb'
output_node_names = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]

with tf.name_scope('input'):
    input_data = tf.placeholder(dtype=tf.float32, shape=(None, input_size, input_size, 3), name='input_data')
model = YOLOV3(input_data, trainable=False)
load_ops = load_weights(tf.global_variables(), darknet_weights)

with tf.Session() as sess:
    sess.run(load_ops)
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        tf.get_default_graph().as_graph_def(),
        output_node_names=output_node_names
    )

    with tf.gfile.GFile(pb_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("{} ops written to {}.".format(len(output_graph_def.node), pb_file))