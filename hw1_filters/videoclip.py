import cv2
import numpy as np
import _G

class VideoClip(cv2.VideoCapture):
  istances = []
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.src = args[0]
    self.frame_max  = int(self.get(cv2.CAP_PROP_FRAME_COUNT))
    self.fps        = self.get(cv2.CAP_PROP_FPS)
    self.resoultion = np.array([int(self.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.get(cv2.CAP_PROP_FRAME_WIDTH))])
    self.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    print(f"Loaded video {self.src}\nfps={self.fps} / {self.frame_max}")
    VideoClip.istances.append(self)

  def frame2timestamp(self, n):
    return n / self.fps

def termiante():
  for cap in VideoClip.istances:
    cap.release()