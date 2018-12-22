import onnx
import torch
from torchvision import transforms
import numpy as np
import pickle
import os
import PIL.Image as Image
from models.MobileNetV2 import MobileNetV2
import onnx_tf.backend as tf_backend
import caffe2.python.onnx.backend as cf2_backend
from onnx_caffe2.backend import Caffe2Backend as cf2_backend2
import tensorflow as tf
# from models import MobileNetV2, BasicModule, BasicModuleSupporter

# ============== Make and save torch model ==============

from config import Config
opt = Config()
net = MobileNetV2(opt)
id2class = pickle.load(open('./ONNX/id_to_class.pkl', 'rb'))
image_path = './ONNX/drawer_square/IMG20181130214623.jpg'

transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

state_dict = torch.load('./source/trained_net/MobileNetV2_Test05_250_NewDataset/best_model.dat',
                        map_location='cpu')['state_dict']
net.load_state_dict(state_dict)
net.eval()
label = image_path.split('/')[-2]

image = Image.open(image_path)
image = transform(image)
image = torch.unsqueeze(image, 0)
print("==== init start ====")
print("The shape of input-image:", image.shape)

_, index = torch.max(net(image), 1)
index = index.item()
same = "True" if id2class[index] == label else "False"
 
print("Test if load model successfully:", same)
print("The test id is {}, class is {}".format(index, id2class[index]))
print("==== init over ====")

# ============== Make onnx-format model ==============

# ==== Bug
# torch.save(net, './ONNX/MobileNetV2_pretrained_model')
# print("Save model successfully")
# Some attributes can't be serialized. Because of Pytorch-basic-module.
# ==== End

dummy_input = image                     # An example of args to input to the network.
input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(314)]
output_names = ["output1"]
torch.onnx.export(net, dummy_input, "./ONNX/MobilenetV2.onnx",
                  input_names=input_names,
                  output_names=output_names,
                  # verbose=True        # It can show the structure of your network.
                  )

print("(Make) ONNX exports successfully.")

# ============== 0nnx to tf ==============
model = onnx.load('./ONNX/MobilenetV2.onnx')
tf_rep = tf_backend.prepare(model)
print("(Make) Make tf_rep successfully.")

np_onnx_image = np.array(image)

with tf.Session() as persisted_sess:
    persisted_sess.graph.as_default()
    tf.import_graph_def(tf_rep.graph.as_graph_def(), name='')
    inp = persisted_sess.graph.get_tensor_by_name(
        tf_rep.tensor_dict[tf_rep.inputs[0]].name
    )
    out = persisted_sess.graph.get_tensor_by_name(
        tf_rep.tensor_dict[tf_rep.outputs[0]].name
    )
    res = persisted_sess.run(out, {inp: np_onnx_image})
    # print(res)

    same = True if id2class[np.argmax(res)] == label else False
    print("(Check) Check if Tf_session predicts right:", same)

# ==== bugs: No normalization, it will be right if you normalize the images.
# onnx_image = Image.open(image_path)
# np_onnx_image = np.array(onnx_image.resize((224, 224)))
# np_onnx_image = np_onnx_image.reshape(3, 224, 224)
# np_onnx_image = np.expand_dims(np_onnx_image, axis=0)
# === end

output = tf_rep.run(np_onnx_image)
same = True if id2class[np.argmax(output)] == label else False
print("(Check) Check if Tf_rep predicts right:", same)

# ============== tf_rep to tf.pb ==============
tf_rep.export_graph('./ONNX/MobilenetV2.pb')
print("(Make) Make tf.pb successfully.")

# ============== ONNX to Caffe2_rep ==============
cf2_rep = cf2_backend.prepare(model)
output = cf2_rep.run(np_onnx_image.astype(np.float32))

same = True if id2class[np.argmax(output)] == label else False
print("(Check) Check if Cf2_rep predicts right:", same)

init_net, predict_net = cf2_backend2.onnx_graph_to_caffe2_net(model.graph)
with open("./ONNX/squeeze_init_net.pb", "wb") as f:
    f.write(init_net.SerializeToString())
with open("./ONNX/squeeze_predict_net.pb", "wb") as f:
    f.write(predict_net.SerializeToString())
print("(Make) Make pbs of caffe2 successfully.")

text = "const char * imagenet_classes[] {" + "\n"
for x in range(len(id2class.keys())):
    text += ('"{}",'.format(id2class[x]) + "\n")
text += "};"
with open("./ONNX/classes.h", "w") as f:
    f.write(text)
print("(Make) Make classes.h(Caffe2) successfully.")