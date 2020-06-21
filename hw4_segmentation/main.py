import scipy.io
import numpy as np
import tensorflow as tf
import cv2
import keras
import re
import pickle as pk
from glob import glob
import matplotlib.pyplot as plt
from keras import backend as K
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from IPython.display import clear_output, display

WINDOW_NAME = "segment"
BATCH_SIZE = 16
EPOCHS     = 30
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

def split_dataset(data):
  np.random.seed(2020)
  idx = np.random.permutation(1000)+1
  training_idx = idx[:750]
  testing_idx  = idx[750:]
  return [training_idx, testing_idx]

def translate_image_files():
  # files = glob("clothing-co-parsing/annotations/pixel-level/*.mat")
  files = glob("labels/*.png")
  ImageSize = (128, 128)

  for file in files:
  #   mat = scipy.io.loadmat(file)
  #   filename = "labels/" + file.split("\\")[-1].split(".")[0] + ".png"
  #   img = cv2.resize(mat['groundtruth'], ImageSize)
  #   img = np.where(img != 0, 0xff, 0)
  #   cv2.imwrite(filename, img)
    print("Resizing", file)
    img = cv2.imread(file)
    img = np.where(img > 0, 1, 0)
    cv2.imwrite(file, img)
# translate_image_files()

def training_image_generator(images):
  for image_file,annotation_file in images:
    yield (image_file,annotation_file)

def load_image(filename,channels=0):
  image = tf.io.read_file(filename)
  if tf.image.is_jpeg(image):
    image = tf.image.decode_jpeg(image,channels=channels)
  else:
    image = tf.image.decode_png(image,channels=channels)
  if channels != 0 and tf.shape(image)[-1]!=channels:
    image = image[:,:,:channels]
  return image

def load_labeled_samples(x,y, augmented = True, preprocessing = None, image_size=(128,128)):
  x_train = tf.image.resize(load_image(x,3), image_size)
  y_train = tf.image.resize(load_image(y,1), image_size,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  if augmented:
    if tf.random.uniform(()) > 0.5:
      x_train = tf.image.flip_left_right(x_train)
      y_train = tf.image.flip_left_right(y_train)
  
  if preprocessing is None:
    x_train = tf.cast(x_train, tf.float32) / 255.0
  else:
    x_train = preprocessing(x_train)

  return (x_train,y_train)

def display_image_mask(display_list,col=8):
  plt.figure(figsize=(2*col, 2*(len(display_list)+col-1)//col))
  for i in range(len(display_list)):
    plt.subplot((len(display_list)+col-1)//col, col, i+1)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis(False)
  plt.tight_layout()
  plt.show()
  return

def upsample(filters, size):
  return [tf.keras.layers.Conv2DTranspose(filters, size, strides=2,padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU()]

def create_prediction_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask

def show_predictions(model, dataset, num=1):
  result = []
  for image, mask in dataset.take(num):
    pred_mask = model.predict(image)
    for i in range(image.shape[0]):
      result.append(image[i])
      result.append(mask[i])
      result.append(create_prediction_mask(pred_mask[i]))
  display_image_mask(result,6)
  return

def plot_history(history, name):
  epochs = range(len(history['loss']))

  plt.figure(figsize=(10,4))
  plt.subplot(1,2,1)
  plt.plot(epochs, history['loss'], label='Training loss')
  plt.plot(epochs, history['val_loss'], label='Validation loss')
  plt.title('Training and Validation Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss Value')
  plt.grid(True)
  plt.legend()
  plt.title('Training/validation loss of {}'.format(name))

  plt.subplot(1,2,2)
  plt.plot(epochs, history['accuracy'], label='Training accuracy')
  plt.plot(epochs, history['val_accuracy'], label='Validation accuracy')
  plt.title('Training and Validation Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.ylim([0, 1])
  plt.legend()
  plt.title('Training/validation accuracy of {}'.format(name))
  plt.grid(True)

  plt.tight_layout()
  plt.show()
  return

def def_unet(classes,height,width,channels,base_model,layer_names,name):
  base_model = tf.keras.Model(inputs=base_model.input,outputs = [base_model.output]+
                                                      [base_model.get_layer(na).output for na in reversed(layer_names)])
  base_model.summary()
  inputs = tf.keras.layers.Input(shape=(height,width,channels))
  
  skips  = base_model(inputs)
  x      = skips[0]
  
  # U型的底部處理
  x      = tf.keras.layers.Conv2D(512,(3,3),padding='same',activation='relu')(x)
  for layer in upsample(512,(3,3)):
    x = layer(x)        
  
  #upsampling network
  for ch,skip in zip([256,128,64,32],skips[1:-1]):
    x = tf.keras.layers.Concatenate()([x, skip])
    for layer in upsample(ch,(3,3)):
      x=layer(x)        
          
  x = tf.keras.layers.Concatenate()([x,skips[-1]]) 
  x = tf.keras.layers.Conv2D(96,(1,1),padding='same')(x)        
  #最後輸出        
  x = tf.keras.layers.Conv2D(classes,(3,3),padding='same')(x)        
  
  unet = tf.keras.Model(inputs=inputs,outputs=x,name=name)
  unet.summary()
  base_model.trainable = False
  unet.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),optimizer='adam',metrics=['accuracy'])
  return unet

regexp = r"\d+"
data_train = np.array(glob("photos/*.jpg"))
data_train = data_train[np.random.permutation(data_train.size)]
training_images, valid_images = train_test_split(np.array([(image_file,f"labels/{re.search(regexp,image_file).group(0)}.png") for image_file in data_train]),test_size=0.1)
preprocess_input(tf.zeros([1, 16, 16, 3]))
dsp_lst = []

train_ds = tf.data.Dataset.from_generator(lambda : training_image_generator(training_images),(tf.string, tf.string)).cache()
train_ds = train_ds.map(lambda x,y: load_labeled_samples(x,y,preprocessing=preprocess_input), num_parallel_calls=tf.data.experimental.AUTOTUNE).take(-1).repeat(EPOCHS).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

valid_ds = tf.data.Dataset.from_generator(lambda : training_image_generator(valid_images),(tf.string, tf.string)).cache()
valid_ds = valid_ds.map(lambda x,y: load_labeled_samples(x,y,augmented=False,preprocessing=preprocess_input), num_parallel_calls=tf.data.experimental.AUTOTUNE).take(-1).repeat(EPOCHS).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

print('training examples:{} validation examples:{}'.format(len(training_images),len(valid_images)))

vgg16_layer_names = [
    'block1_conv2', # 1
    'block2_conv2', # 1/2
    'block3_conv3', # 1/4
    'block4_conv3', # 1/8
    'block5_conv3', # 1/16
]

unet_vgg16 = def_unet(2, IMAGE_WIDTH, IMAGE_HEIGHT, 3, tf.keras.applications.VGG16(include_top = False),vgg16_layer_names,'unet-vgg16')
# tf.keras.utils.plot_model(unet_vgg16, show_shapes=True)

VALIDATION_STEPS = len(valid_images)//BATCH_SIZE
STEPS_PER_EPOCH  = len(training_images)//BATCH_SIZE
unet_vgg16_history = unet_vgg16.fit(train_ds, epochs=EPOCHS,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_steps=VALIDATION_STEPS,
                        validation_data=valid_ds,
                        callbacks=[])#[DisplayCallback()])


plot_history(unet_vgg16_history.history, 'unet-vgg16')
show_predictions(unet_vgg16,valid_ds)

with open("unet_vgg16.net", "wb") as fp:
  pk.dump(unet_vgg16, fp)