import tensorflow as tf
import cv2 
import PIL
from tensorflow.keras import Model
from tensorflow.python.eager.backprop import GradientTape


import matplotlib.pyplot as plt
import numpy as np

### Model 
#vgg16 = tf.keras.applications.VGG16(include_top=True,weights="ImageNet")

contents = ['block4_conv2']
styles = ["block1_conv1","block2_conv1","block3_conv1","block4_conv1","block5_conv1"]

opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.01,beta_1=0.99, epsilon=1e-1)




def gram_matrix(layer):
  result = tf.linalg.einsum("lijc,lijd->lcd",layer,layer)
  input_shape = tf.shape(layer);
  return (result/(tf.cast((input_shape[0] * input_shape[1]) , tf.float32)))
  # return (result/(input_shape[0] * input_shape[1]))


def load_model():
    vgg = tf.keras.applications.VGG16(include_top=True,weights= 'imagenet')
    #vgg.load_weights("/home/arvin/Desktop/p3r50n47/n/vgg16_weights_tf_dim_ordering_tf_kernels.h5")
    vgg.trainable = False

    content_output = vgg.get_layer(contents[0]).output

    style_outputs = [vgg.get_layer(style).output for style in styles]

    gram_output = [gram_matrix(style_output) for style_output in style_outputs]

    return Model([vgg.input],[content_output,style_outputs])


def loss(style_output , content_output , content_target , style_target):
  cnt_weight = 0.8
  sty_weight = 1e-1

  content_loss = tf.reduce_mean((content_output - content_target)**2)

  style_loss = tf.add_n([tf.reduce_mean((output_ - target_)**2) for output_ , target_ in zip(style_output , style_target)])

  return (cnt_weight * content_loss) + (sty_weight * style_loss)


def train(vgg_model, image , epoch , content_target, style_target):
  with GradientTape() as tape:
    output = vgg_model(image * 255)
    t_loss = loss(output[1] , output[0] , content_target , style_target)
  gradient = tape.gradient(t_loss , image)
  opt.apply_gradients([(gradient , image)])
  image.assign(tf.clip_by_value(image, clip_value_min = 0, clip_value_max = 1 ))

  tf.print(t_loss)



def generate(content_image, style_image):
  
    vgg_model = load_model()

    content_target = vgg_model(np.array([content_image]))[0]
    style_target = vgg_model(np.array([style_image]))[1]

    result_image = tf.image.convert_image_dtype(content_image , tf.float32)  

    result = tf.Variable([result_image])

    Epoch = 10

    for i in range(Epoch):
        train(vgg_model, result , i , content_target, style_target)

    result = result * 255

    tensor = result
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
    tensor = tensor[0]
    tensor =  PIL.Image.fromarray(tensor)

    return tensor

    #plt.imshow(cv2.cvtColor(np.array(tensor), cv2.COLOR_BGR2RGB))
