import os
import cv2
import numpy as np
import _G
import videoclip
from videoclip import VideoClip
from videoplayer import VideoPlayer


def test():
  blank = np.zeros((720, 1280, 3), np.uint8)
  blank[0:240,:] = (0xff,0,0)
  blank[240:480,:] = (0,0xff,0)
  blank[480:720,:] = (0,0,0xff)
  cv2.imshow("test", blank)
  cv2.waitKey(0)
  exit()

def attach_audio(afile, vfile):
  os.system(f"ffmpeg -i {vfile} -i {afile} -c copy -map 0:v:0 -map 1:a:0 final_{vfile}")

# test()

main_video = VideoClip(_G.SourceFilename)
player = VideoPlayer(main_video, _G.TrackbarName, _G.SourceFilename)

def main_loop():
  player.start()
  while not _G.FLAG_STOP and not player.is_ended():
    player.update()

def terminate():
  videoclip.termiante()
  cv2.destroyAllWindows()

main_loop()
terminate()

vfile = _G.make_out_filename(player.src)
afile = _G.make_audio_filename(player.src)
attach_audio(afile, vfile)