import os.path
from queue import Queue
from threading import Thread
import cv2
import moviepy.editor as mp
import numpy as np
from ffpyplayer.player import MediaPlayer
import _G
from _G import make_audio_filename, make_out_filename
import filter

TrackBarName   = 'frame no.'
WindowName     = "QuadraOcean"

class VideoPlayer:
  
  def __init__(self, video, trackbar_name, window_name):
    self.cur_frame = 0
    self.src       = video.src
    self.video     = video
    self.audio     = MediaPlayer(video.src)
    self.frame_max = video.frame_max
    self.trackbar  = trackbar_name
    self.window    = window_name
    self.ostream   = self.init_ostream()
    self.queue     = Queue(maxsize=_G.MaxQueueSize)
    self.FLAG_CODEC_STOP = False
    cv2.namedWindow(self.window)
    cv2.createTrackbar(self.trackbar, self.window, 0, self.frame_max, self.set_current_frame)
  
  def init_ostream(self):
    fname  = make_out_filename(self.video.src)
    fourcc = cv2.VideoWriter_fourcc(*_G.VideoCodec)
    _fps   = self.video.fps
    _res   = (_G.CanvasWidth, _G.CanvasHeight)
    return cv2.VideoWriter(fname, fourcc, _fps, _res)

  def set_current_frame(self, n):
    # self.cur_frame = n
    pass

  def set_audio_frame(self, n):
    t = self.video.frame2timestamp(n)
    self.audio.seek(t, False)

  def start(self):
    self.codec_t = Thread(target=self.update_codec)
    self.codec_t.daemon = True
    self.codec_t.start()
    _t = Thread(target=self.extract_audio)
    _t.start()
    return self

  def extract_audio(self):
    fname = make_audio_filename(self.src)
    if not os.path.exists(fname):
      v = mp.VideoFileClip(self.src)
      v.audio.write_audiofile(fname)

  def update_codec(self):
    while not self.FLAG_CODEC_STOP:
      if not self.queue.full():
        ret, frame = self.video.read()
        if not ret:
          self.FLAG_CODEC_STOP = True
          return
        frame = self.make_frame(frame)
        self.queue.put(frame)
    print("Codec Ended")

  def frame_available(self):
    return self.queue.qsize() > 0

  def get_frame(self):
    return self.queue.get()

  def update(self):
    self.update_frame()
    self.update_input()
  
  def update_frame(self):
    if self.is_ended() or _G.FLAG_PAUSE:
      return
    
    cv2.setTrackbarPos(self.trackbar, self.window, self.cur_frame)
    
    frame = self.get_frame()
    if frame is None:
      return
      
    cv2.imshow(self.window, frame)
    # print(f"qsize={self.queue.qsize()}")
    self.ostream.write(frame)

    if not _G.FLAG_PAUSE:
      self.cur_frame += 1

  def update_input(self):
    key = cv2.waitKey(_G.UPS)
    if key == _G.VK_ESC:
      _G.FLAG_STOP = True
    elif key == _G.VK_SPACE:
      _G.FLAG_PAUSE ^= True
      self.audio.toggle_pause()

  def is_ended(self):
    return self.cur_frame >= self.frame_max
  
  def make_audio_window(self):
    window, val = self.audio.get_frame()
    if window is None or val == 'eof':
      return (None,None)
    return window

  def make_frame(self, frame):
    canvas = np.zeros((_G.CanvasHeight, _G.CanvasWidth, 3), np.uint8)

    mx, my = _G.CanvasWidth // 2, _G.CanvasHeight // 2
    frame = cv2.resize(frame, (mx, my))

    frame2 = filter.greyscale(frame)
    frame3 = filter.sharpen(frame)
    frame4 = filter.inverted(frame)
    
    canvas[0:frame.shape[0], 0:frame.shape[1]] += frame
    canvas[0:frame.shape[0], mx:mx+frame.shape[1]] += frame2
    canvas[my:my+frame.shape[0], 0:frame.shape[1]] += frame3
    canvas[my:my+frame.shape[0], mx:mx+frame.shape[1]] += frame4
    return canvas
