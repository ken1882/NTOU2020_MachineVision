import cv2
import numpy as np
from glob import glob
from copy import copy, deepcopy
from collections import defaultdict

DATA_FOLDER = "dataset1"
IMG_FORMAT  = "jpg"
WINDOW_NAME = "stitch"

ORI_WIDTH, ORI_HEIGHT = 1920, 1080
RESOULTION   = np.array([ORI_WIDTH, ORI_HEIGHT])

DIV_FACTOR = 4
MUL_FACTOR = 3

CANVAS_SIZE  = np.array([RESOULTION[0] // DIV_FACTOR, RESOULTION[1] // DIV_FACTOR * MUL_FACTOR])
DISPLAY_SIZE = RESOULTION // DIV_FACTOR

FeatureDetector = cv2.xfeatures2d.SIFT_create()
FeatureMatcher  = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

CANVAS_WIDTH  = ORI_WIDTH // DIV_FACTOR * MUL_FACTOR
CANVAS_HEIGHT = ORI_HEIGHT // DIV_FACTOR
VideoCanvas  = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3))
CanvasCount  = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH))
CanvasOnes   = np.ones((CANVAS_HEIGHT, CANVAS_HEIGHT))

CANDIDATE_NUM = 5

METHOD_HOMOGRAPHY = 0
METHOD_STACKING   = 1
METHOD_ADD        = 2
METHOD_FILLING    = 3

def get_img_size(img):
  return [img.shape[1], img.shape[0]]

def load_and_resize(img):
  return cv2.resize(cv2.imread(img), tuple(DISPLAY_SIZE))

def sort_by_findex(fname):
  return int(fname.split(f".{IMG_FORMAT}")[0].split('_')[-1])

def merge_matrix(big, small, st_r, st_c, op='='):
  st_r = big.shape[0] + st_r if st_r < 0 else st_r
  st_c = big.shape[1] + st_r if st_c < 0 else st_c
  ed_r, ed_c = st_r+small.shape[0], st_c+small.shape[1]
  if op == '=':
    big[st_r:ed_r, st_c:ed_c] = small
  elif op == '+':
    big[st_r:ed_r, st_c:ed_c] += small
  elif op == 'x':
    sub = big[st_r:ed_r, st_c:ed_c]
    big[st_r:ed_r, st_c:ed_c] = np.where(sub == 0, small, sub)

class ImageFragment:

  def __init__(self, file, dx=None, dy=None):
    self.file = file
    self.dx   = dx
    self.dy   = dy
    self.next = None
    self.next_score = 0

  def load(self):
    return load_and_resize(self.file)


def stitich_ordered(files, method, offset):
  global VideoCanvas, CanvasCount, CanvasOnes
  images = []
  img0  = load_and_resize(files[0])
  images.append(img0)
  gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
  kp1 = FeatureDetector.detect(gray0,None)
  dt1 = FeatureDetector.compute(gray0,kp1)[1]
  T   = np.eye(3)
  T[0,2] = offset
  T[1,2] = 0
  VideoCanvas = cv2.warpPerspective(img0, T, (CANVAS_WIDTH, CANVAS_HEIGHT)).astype(np.float)

  fragments = []
  frag0 = ImageFragment(files[0])

  if method == METHOD_HOMOGRAPHY:
    t_count = cv2.warpPerspective(CanvasOnes, T, (CANVAS_WIDTH, CANVAS_HEIGHT)).astype(np.float)
    CanvasCount += t_count.astype(np.float)
  elif method == METHOD_STACKING:
    curx = offset
    errorx = 0.0
  print(curx)
  for i, file in enumerate(files):
    if i == 0:
      continue
    img2 = load_and_resize(file)
    images.append(img2)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp2, dt2 = FeatureDetector.detectAndCompute(gray2, None)
    matches = sorted(FeatureMatcher.match(dt2, dt1), key=lambda x:x.distance)
    
    if method == METHOD_HOMOGRAPHY:
      src, dst = [], []
      for m in matches:
        src.append(kp2[m.queryIdx].pt + (1,))
        dst.append(kp1[m.trainIdx].pt + (1,))
      src = np.array(src,dtype=np.float)
      dst = np.array(dst,dtype=np.float)
      print(f"Matches: {len(matches)}")
      print(f"{kp2[matches[0].queryIdx].pt}")
      print(f"{src[0]} {dst[0]}")
      print(src)
      print('-'*15)
      print(dst)
      print('='*15)
      A, mask = cv2.findHomography(src, dst, cv2.RANSAC)
      print(A)
      print('-'*15)
      print(mask.shape)
      T = T.dot(A)
      warp_img = cv2.warpPerspective(img2,T,(CANVAS_WIDTH, CANVAS_HEIGHT)).astype(np.float)
      t_count  = cv2.warpPerspective(CanvasOnes,T,(CANVAS_WIDTH, CANVAS_HEIGHT)).astype(np.float)
      VideoCanvas += warp_img
      CanvasCount += t_count.astype(np.float)
      t_count = CanvasCount.copy()
      t_count[t_count == 0] = 1
      disp = VideoCanvas.copy()
      disp[:,:,0] = VideoCanvas[:,:,0] / t_count
      disp[:,:,1] = VideoCanvas[:,:,1] / t_count
      disp[:,:,2] = VideoCanvas[:,:,2] / t_count
      cv2.imshow(WINDOW_NAME, np.array(disp, dtype=np.uint8))
      cv2.imshow('matching',cv2.drawMatches(img2,kp2,img0,kp1,matches[:CANDIDATE_NUM], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS))
    elif method == METHOD_STACKING:
      dx = 0
      for i, m in enumerate(matches):
        if i > CANDIDATE_NUM:
          break
        delta = np.array(kp2[m.queryIdx].pt) - np.array(kp1[m.trainIdx].pt)
        dx += delta[0]
      dx /= CANDIDATE_NUM
      errorx += dx - int(dx)
      if errorx > 1.0 or errorx < -1.0:
        dx += int(errorx)
        errorx -= int(errorx)
      curx = curx - int(dx)
      
      frag1 = ImageFragment(file, dx, 0)
      frag0.next = frag1
      fragments.append(frag0)
      frag0 = frag1

      print(f"{dx} {curx} {errorx}")
      print(img2.shape)
      merge_matrix(VideoCanvas, img2, 0, curx)
      cv2.imshow(WINDOW_NAME, np.array(VideoCanvas, dtype=np.uint8))
      cv2.imshow('matching',cv2.drawMatches(img2,kp2,img0,kp1,matches[:CANDIDATE_NUM], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS))

    key = cv2.waitKey(20) & 0xFF
    if key == 27:
      break
    img0 = img2
    kp1  = kp2
    dt1  = dt2
  
  if method == METHOD_STACKING:
    for frag in fragments:
      if frag.next:
        print(f"{frag.file} -> {frag.next.file} ({frag.next.dx}, {frag.next.dy})")
    generate_final_image(fragments, images)

def generate_final_image(fragments, images=[], method=METHOD_STACKING):
  print("Assemble image fragments...")
  sx = sy = 0
  frag = fragments[0]
  flag_load = not images
    
  if flag_load:
    images.append(frag.load())
  else:
    img = images[0]
  fy, fx = img.shape[0:2]
  dr_x = [0]
  dr_y = [0]
  lx = ly = 0
  idx = 1
  while frag.next:
    if flag_load:
      img = frag.next.load()
      images.append(img)
    else:
      img = images[idx]
    h, w = img.shape[0:2]
    frag = frag.next
    sx = min(sx, lx - frag.dx)
    sy = min(sy, ly - frag.dy)
    fx = max(fx, lx - frag.dx + w)
    fy = max(fy, ly - frag.dy + h)
    lx -= frag.dx
    ly -= frag.dy
    dr_x.append(lx)
    dr_y.append(ly)
    print(f"BB: {sx} {sy} {fx} {fy}")
    print(f"Cur: {lx} {ly}")
    idx += 1
  
  print(dr_x)
  print(dr_y)
  errx = erry = 0
  curx = cury = 0
  shiftx = int(-sx)
  shifty = int(-sy)
  fx = int(fx + shiftx + 1)
  fy = int(fy + shifty + 1)
  canvas = np.zeros((fy, fx, 3))
  if method == METHOD_ADD:
    times = np.zeros((fy, fx), dtype=np.int32)
  print(f"Final resoultion: {fx}*{fy}")
  idx = 0
  for x, y in zip(dr_x, dr_y):
    errx += x - int(x)
    erry += y - int(y)
    dx = int(x + shiftx)
    dy = int(y + shifty)
    if errx > 1.0 or errx < -1.0:
      dx += int(errx)
      errx -= int(errx)
    if erry > 1.0 or erry < -1.0:
      dy += int(erry)
      erry -= int(erry)
    op = '='
    if method == METHOD_ADD:
      op = '+'
    elif method == METHOD_FILLING:
      op = 'x'
    merge_matrix(canvas, images[idx], dy, dx, op=op)
    if method == METHOD_ADD:
      ones = np.ones(images[idx].shape[0:2], dtype=np.int32)
      merge_matrix(times, ones, dy, dx, op='+')
    idx += 1
  if method == METHOD_ADD:
    times = np.where(times == 0, 1, times)
    canvas[:,:,0] /= times
    canvas[:,:,1] /= times
    canvas[:,:,2] /= times
  cv2.imshow(WINDOW_NAME, np.array(canvas, np.uint8))
  cv2.waitKey(0)

def stitich_unordered(files, method):
  global VideoCanvas, CanvasCount, CanvasOnes
  images = []
  kps = []
  dts = []
  img0  = load_and_resize(files[0])
  images.append(img0)
  gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
  kp1 = FeatureDetector.detect(gray0,None)
  kps.append(kp1)
  dt1 = FeatureDetector.compute(gray0,kp1)[1]
  dts.append(dt1)
  T   = np.eye(3)
  T[0,2] = 0
  T[1,2] = 0
  likelihood = defaultdict(list)
  fragments = [ImageFragment(files[0])]
  for i, file in enumerate(files):
    if i == 0:
      continue
    fragments.append(ImageFragment(file))
    img2 = load_and_resize(file)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp2, dt2 = FeatureDetector.detectAndCompute(gray2, None)
    print('-'*20)
    for idx, img0 in enumerate(images):
      kp1 = kps[idx]
      dt1 = dts[idx]
      matches = sorted(FeatureMatcher.match(dt2, dt1), key=lambda x:x.distance)
      cv2.imshow('matching',cv2.drawMatches(img2,kp2,img0,kp1,matches[:CANDIDATE_NUM], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS))
      tops = matches[0:CANDIDATE_NUM]
      score0 = len(matches)
      score1 = -sum(m.distance for m in tops)
      print(f"Matched: {score0} {score1} {score0+score1}")
      likelihood[idx].append((len(images), score0+score1, copy(kp1), copy(kp2), tops))
      key = cv2.waitKey(20)
      if key == 27:
        return
    kps.append(kp2)
    dts.append(dt2)
    images.append(img2)
  for i in likelihood:
    likelihood[i] = sorted(likelihood[i], key=lambda x:-x[1])
    print('-'*15)
    print(i)
    fragments[i].next = fragments[likelihood[i][0][0]]
    for v in likelihood[i]:
      print(v[0], v[1])
  
  for i, frag in enumerate(fragments):
    print(f"{frag.file} -> {frag.next.file if frag.next else 'EOF'}")
    if i >= len(likelihood):
      break
    info = likelihood[i][0]
    kp1  = info[2]
    kp2  = info[3]
    matches = info[4]
    dx = dy = 0
    for m in matches:
      delta = np.array(kp2[m.queryIdx].pt) - np.array(kp1[m.trainIdx].pt)
      dx += delta[0]
      dy += delta[1]
    dx //= CANDIDATE_NUM
    dy //= CANDIDATE_NUM
    print(f"Delta: {dx} {dy}")
    frag.next.dx = dx
    frag.next.dy = dy
  generate_final_image(fragments, images)
  generate_final_image(fragments, images, method=METHOD_ADD)
  generate_final_image(fragments, images, method=METHOD_FILLING)

files = sorted(glob(f"{DATA_FOLDER}/*.{IMG_FORMAT}"), key=sort_by_findex)
stitich_ordered(files[::-1], METHOD_STACKING, 0)
files = glob(f"dataset2/*.JPG")
stitich_unordered(files, METHOD_STACKING)
cv2.destroyAllWindows()