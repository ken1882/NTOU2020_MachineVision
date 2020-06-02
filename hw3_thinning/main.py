import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import copy

def _match_transition(mat_ele):
  val = mat_ele
  fst = val & 1
  lst = fst
  cur = 0
  val = val >> 1
  cnt = 0
  while val > 0:
    cur = val & 1
    if lst == 0 and cur == 1:
      cnt += 1
    lst = cur
    val = val >> 1
  if cur == 0 and fst == 1:
    cnt += 1
  return cnt == 1

match_transition = np.vectorize(_match_transition)
band = tf.bitwise.bitwise_and

###
class AdaptiveThreshold:
  def __init__(self,blockSize,C):
    self.blockSize = blockSize
    self.filters = tf.ones((blockSize,blockSize,1,1),dtype=tf.float32)/blockSize**2
    self.C       = tf.constant(-C, dtype=tf.float32)
  
  def __call__(self, ori_input, pad_input):
    x  = tf.nn.conv2d(pad_input, self.filters, strides=1, padding="VALID")
    th = x - self.C
    # return the resultant image, where 1 represents above the threshold and 0 represents below the threshold
    return np.where(ori_input > th, 1, 0)
    
class Thinning:
  
  def __init__(self):
    self.filters1, self.filters2 = self._surface_patterns() 

  @staticmethod
  def _surface_patterns():
    # generate the filters
    filters1 = np.array([
      [1, 1, 1],
      [1, 0, 1],
      [1, 1, 1]
    ]).reshape(3, 3, 1, 1)
    filters2 = np.array([
      [  1,  2,  4],
      [128,  0,  8],
      [ 64, 32, 16]
    ]).reshape(3, 3, 1, 1)
    return filters1, filters2

  def __call__(self, inputs):
    #  do thinning
    #  padding is required
    x_last = copy(inputs)
    cnt = 0
    x   = copy(inputs)
    while True:
      print(f"Thinning no.{cnt} times")
      # pass 1
      x2 = tf.nn.conv2d(x, self.filters1, strides=1, padding="SAME")
      x3 = tf.nn.conv2d(x, self.filters2, strides=1, padding="SAME")
      x2 = np.where(((x2 >= 2) & (x2 <= 6)), 1, 0)
      x3 = np.where(
        (
          (
            (
              ((band(x3,2) == 0) | (band(x3,8) == 0) | (band(x3,32) == 0)) &
              ((band(x3,128) == 0) | (band(x3,8) == 0) | (band(x3,32) == 0))
            )
          ) & match_transition(x3)
        ), 1, 0
      )
      x = np.where(((x == 1) & (x2 == 1) & (x3 == 1)), 0, x)
      # pass 2
      x2 = tf.nn.conv2d(x, self.filters1, strides=1, padding="SAME")
      x3 = tf.nn.conv2d(x, self.filters2, strides=1, padding="SAME")
      x2 = np.where(((x2 >= 2) & (x2 <= 6)), 1, 0)
      x3 = np.where(
        (
          (
            (
              ((band(x3,2) == 0) | (band(x3,8) == 0) | (band(x3,128) == 0)) &
              ((band(x3,2) == 0) | (band(x3,32) == 0) | (band(x3,128) == 0))
            )
          ) & match_transition(x3)
        ), 1, 0
      )
      x = np.where(((x == 1) & (x2 == 1) & (x3 == 1)), 0, x)
      cnt += 1
      # if no pixels are changed, break this loop
      if np.all(x == x_last):
        break
      x_last = x.copy()
      img = x[:,1:-1,1:-1,:]
      img = tf.where(tf.squeeze(x)>0,tf.constant([[255]],dtype=tf.uint8),tf.constant([[0]],dtype=tf.uint8))
      cv2.imshow("thin",img.numpy())
      cv2.waitKey(20)
    cv2.destroyAllWindows()
    outputs = x[:,1:-1,1:-1,:]  
    print(f"Looped {cnt} times")
    return outputs


#下載測試影像
# url      = 'https://evatronix.com/images/en/offer/printed-circuits-board/Evatronix_Printed_Circuits_Board_01_1920x1080.jpg'
# testimage= tf.keras.utils.get_file('pcb.jpg',url)


#讀入測試影像
inputs   = cv2.imread('pcb.jpg')
# inputs = cv2.imread('test.png')
print(inputs.shape)

#轉成灰階影像
inputs   = cv2.cvtColor(inputs,cv2.COLOR_BGR2GRAY)

#顯示測試影像
plt.figure(figsize=(20,15))
plt.imshow(inputs,cmap='gray')
plt.axis(False)
plt.show()

#轉換影像表示方式成四軸張量(sample,height,width,channel)，以便使用卷積運算。
inputs = tf.convert_to_tensor(inputs,dtype=tf.float32)
PADDING_WIDTH = 30
paddings = tf.constant([[PADDING_WIDTH, PADDING_WIDTH], [PADDING_WIDTH, PADDING_WIDTH]])
pad_inputs = copy(inputs)
pad_inputs = tf.pad(pad_inputs, paddings, "SYMMETRIC")
pad_inputs = pad_inputs[tf.newaxis,:,:,tf.newaxis]
inputs = inputs[tf.newaxis,:,:,tf.newaxis]

#使用卷積運算製作AdatpiveThresholding
binary = AdaptiveThreshold(61,-8)(inputs, pad_inputs)

#存下AdaptiveThresholding結果
outputs = tf.where(tf.squeeze(binary)>0,tf.constant([[255]],dtype=tf.uint8),tf.constant([[0]],dtype=tf.uint8))
cv2.imwrite('pcb_threshold.png',outputs.numpy())

#顯示AdaptiveThresholding結果
cv2.imshow("threshold", np.array(tf.squeeze(binary).numpy()*255, dtype=np.uint8))
cv2.waitKey(0)

outputs = tf.where(tf.squeeze(Thinning()(binary))>0,tf.constant([[255]],dtype=tf.uint8),tf.constant([[0]],dtype=tf.uint8))
outputs = tf.squeeze(outputs)

#存下細線化結果
cv2.imwrite('pcb_thinning.png',outputs.numpy())

#注意由於螢幕解析度，同學在螢幕上看到的細線化結果可能不是真正結果，此時必須看存下來的結果影像。
plt.figure(figsize=(20,15))        
plt.imshow(outputs.numpy(),cmap='gray')
plt.axis(False)
plt.show()