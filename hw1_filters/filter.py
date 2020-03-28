import _G
import tensorflow as tf
import numpy as np
import cv2

GREY_KERNEL = np.array((0.25, 0.3, 0.25))
GREY_KERNEL = GREY_KERNEL.reshape(1, 1, 3, 1)
TF_GERY_KERNEL = tf.convert_to_tensor(GREY_KERNEL, tf.float32)

def translate_bgr_kernel(h, w, ink):
  tmp = np.zeros((h, w, 3, 3))
  for i in range(h):
    for j in range(w):
      tmp[i,j,2,2] += (ink[i,j] / h / w)
  # tmp[2,2,:,:] = ink
  return tmp

SHARP_KERNEL = np.array((
  ( 0, -1,  0),
  (-1,  5, -1),
  ( 0, -1,  0),
))

EMBOSS_KERNEL = np.array((
  (-2, -1, 0),
  (-1,  1, 1),
  ( 0,  1, 2),
))

TF_SHARP_KERNEL  = tf.convert_to_tensor(translate_bgr_kernel(*(SHARP_KERNEL.shape),SHARP_KERNEL), tf.float32)
TF_EMBOSS_KERNEL = tf.convert_to_tensor(translate_bgr_kernel(*(EMBOSS_KERNEL.shape),EMBOSS_KERNEL), tf.float32)

def fit_tfdim(data):
  if data.ndim >= 4:
    return data
  return data[np.newaxis, ...]

def greyscale(frame):
  ow, oh = frame.shape[0:2]
  frame = tf.nn.conv2d(fit_tfdim(frame), TF_GERY_KERNEL, _G.NN_STRIDE, _G.NN_PADDING)
  frame = np.array(np.squeeze(frame), dtype=np.uint8)
  return np.repeat(frame, 3).reshape(ow, oh, 3)

def process_V_filter(frame, kernel):
  oritf = np.array(frame / 0xFF, dtype=np.float32)
  tmp = tf.image.rgb_to_hsv(oritf[..., ::-1])
  tmp = tf.nn.conv2d(fit_tfdim(tmp), kernel, _G.NN_STRIDE, _G.NN_PADDING)
  tmp = np.array(tf.image.hsv_to_rgb(tmp))
  tmp = np.clip(oritf + np.squeeze(tmp)[..., ::-1], 0, 1)
  return np.array(tmp * 0xff, dtype=np.uint8)

def sharpen(frame):
  return process_V_filter(frame, TF_SHARP_KERNEL)

def emboss(frame):
  return process_V_filter(frame, TF_EMBOSS_KERNEL)

if __name__ == "__main__":
  img = cv2.imread("test.png")
  print(img, img.shape)
  print('-'*10)
  img2 = sharpen(img)
  print(img2, img2.shape)
  print(img == img2)
  cv2.imshow("test", img2)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
