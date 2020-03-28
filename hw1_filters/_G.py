import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def make_out_filename(ori):
  return "out_" + "".join(ori.split('.')[:-1]) + ".mkv"

def make_audio_filename(ori):
  return "".join(ori.split('.')[:-1]) + ".mp3"

SourceFilename = "test_video.mp4"
TrackbarName   = "frame no."

VK_ESC   = 27
VK_SPACE = 32

UPS = int(1000 / 120)

VideoCodec = 'h264'

CanvasHeight = 900
CanvasWidth  = 1600

MaxQueueSize = 128

NN_PADDING = 'SAME'
NN_STRIDE  = 1

FLAG_STOP  = False
FLAG_PAUSE = False