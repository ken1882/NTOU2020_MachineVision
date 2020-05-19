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
CANVAS_WIDTH2  = int(CANVAS_WIDTH  * 1.2)
CANVAS_HEIGHT2 = int(CANVAS_HEIGHT * 4)

VideoCanvas   = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3))
VideoCanvas2  = np.zeros((CANVAS_HEIGHT2, CANVAS_WIDTH2, 3))

CanvasCount   = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH))
CanvasCount2   = np.zeros((CANVAS_HEIGHT2, CANVAS_WIDTH2))

CANDIDATE_NUM = 5

METHOD_HOMOGRAPHY = 0
METHOD_STACKING   = 1
METHOD_ADD        = 2
METHOD_FILLING    = 3

WARP_FLAGS = cv2.INTER_AREA+cv2.WARP_FILL_OUTLIERS

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

def stitich_ordered_homography(files, current_canvas):
  global VideoCanvas, VideoCanvas2
  if current_canvas.shape == VideoCanvas.shape:
    cur_canvas_width  = CANVAS_WIDTH
    cur_canvas_height = CANVAS_HEIGHT
    cur_counter = CanvasCount
  elif current_canvas.shape == VideoCanvas2.shape:
    cur_canvas_width  = CANVAS_WIDTH2
    cur_canvas_height = CANVAS_HEIGHT2
    cur_counter = CanvasCount2

  lobound, upbound = 0, len(files) - 1
  mid = upbound // 2
  img0 = load_and_resize(files[mid])
  gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
  kp1,dt1 = FeatureDetector.detectAndCompute(gray, None)
  kp0,dt0 = kp1,dt1
  T1      = np.eye(3)
  T1[0,2] = (cur_canvas_width - img0.shape[1]) // 2
  T1[1,2] = 0
  print(cur_canvas_width, img0.shape[1], T1[0,2])
  ones = np.ones(tuple(DISPLAY_SIZE[::-1]))
  current_canvas = cv2.warpPerspective(img0, T1, (cur_canvas_width, cur_canvas_height), flags=WARP_FLAGS).astype(np.float)
  t_count  = cv2.warpPerspective(ones,T1,(cur_canvas_width, cur_canvas_height), flags=WARP_FLAGS).astype(np.float)
  cur_counter += t_count.astype(np.float)
  cv2.imshow(WINDOW_NAME, np.array(current_canvas, dtype=np.uint8))
  cv2.waitKey(0)
  idx = mid + 1
  while idx <= upbound:
    img2 = load_and_resize(files[idx])
    print(files[idx])
    kp2,dt2 = FeatureDetector.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
    matches = sorted(FeatureMatcher.match(dt2, dt1), key=lambda x:x.distance)
    src, dst = [], []
    for m in matches:
      src.append(kp2[m.queryIdx].pt + (1,))
      dst.append(kp1[m.trainIdx].pt + (1,))
    src = np.array(src,dtype=np.float)
    dst = np.array(dst,dtype=np.float)
    A, mask = cv2.findHomography(src, dst, cv2.RANSAC)
    T1 = T1.dot(A)
    warp_img = cv2.warpPerspective(img2,T1,(cur_canvas_width, cur_canvas_height), flags=WARP_FLAGS).astype(np.float)
    t_count  = cv2.warpPerspective(ones,T1,(cur_canvas_width, cur_canvas_height), flags=WARP_FLAGS).astype(np.float)
    current_canvas += warp_img
    cur_counter += t_count.astype(np.float)
    t_count = cur_counter.copy()
    t_count[t_count == 0] = 1
    disp = current_canvas.copy() 
    disp[:,:,0] = current_canvas[:,:,0] / t_count
    disp[:,:,1] = current_canvas[:,:,1] / t_count
    disp[:,:,2] = current_canvas[:,:,2] / t_count
    cv2.imshow(WINDOW_NAME, np.array(disp, dtype=np.uint8))
    cv2.imshow('matching',cv2.drawMatches(img2,kp2,img0,kp1,matches[:CANDIDATE_NUM], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS))
    key = cv2.waitKey(20)
    if key == 27:
      return
    idx += 1
    img0 = img2
    kp1  = kp2
    dt1  = dt2
  T1      = np.eye(3)
  T1[0,2] = (cur_canvas_width - img0.shape[1]) // 2
  T1[1,2] = 0
  kp1,dt1 = kp0,dt0
  idx = mid - 1
  while idx >= lobound:
    img2 = load_and_resize(files[idx])
    print(files[idx])
    kp2,dt2 = FeatureDetector.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
    matches = sorted(FeatureMatcher.match(dt2, dt1), key=lambda x:x.distance)
    src, dst = [], []
    for m in matches:
      src.append(kp2[m.queryIdx].pt + (1,))
      dst.append(kp1[m.trainIdx].pt + (1,))
    src = np.array(src,dtype=np.float)
    dst = np.array(dst,dtype=np.float)
    A, mask = cv2.findHomography(src, dst, cv2.RANSAC)
    T1 = T1.dot(A)
    warp_img = cv2.warpPerspective(img2,T1,(cur_canvas_width, cur_canvas_height), flags=WARP_FLAGS).astype(np.float)
    t_count  = cv2.warpPerspective(ones,T1,(cur_canvas_width, cur_canvas_height), flags=WARP_FLAGS).astype(np.float)
    current_canvas += warp_img
    cur_counter += t_count.astype(np.float)
    t_count = cur_counter.copy()
    t_count[t_count == 0] = 1
    disp = current_canvas.copy() 
    disp[:,:,0] = current_canvas[:,:,0] / t_count
    disp[:,:,1] = current_canvas[:,:,1] / t_count
    disp[:,:,2] = current_canvas[:,:,2] / t_count
    cv2.imshow(WINDOW_NAME, np.array(disp, dtype=np.uint8))
    cv2.imshow('matching',cv2.drawMatches(img2,kp2,img0,kp1,matches[:CANDIDATE_NUM], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS))
    key = cv2.waitKey(20)
    if key == 27:
      return
    idx -= 1
    img0 = img2
    kp1  = kp2
    dt1  = dt2
  cv2.waitKey(0)

def stitich_ordered(files, method, offset):
  global VideoCanvas, CanvasCount
  images = []
  img0  = load_and_resize(files[0])
  images.append(img0)
  gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
  kp1,dt1 = FeatureDetector.detectAndCompute(gray0, None)
  T      = np.eye(3)
  T[0,2] = offset
  T[1,2] = 0
  VideoCanvas = cv2.warpPerspective(img0, T, (CANVAS_WIDTH, CANVAS_HEIGHT), flags=WARP_FLAGS).astype(np.float)
  fragments = []
  frag0 = ImageFragment(files[0])
  
  if method == METHOD_STACKING:
    curx = offset
    errorx = 0.0
  
  for i, file in enumerate(files):
    if i == 0:
      continue
    img2 = load_and_resize(file)
    images.append(img2)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp2, dt2 = FeatureDetector.detectAndCompute(gray2, None)
    matches = sorted(FeatureMatcher.match(dt2, dt1), key=lambda x:x.distance)
    
    if method == METHOD_STACKING:
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
    dx = max(0, dx)
    dy = max(0, dy)
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

def stitich_pizza(files):
  global VideoCanvas2, CanvasCount2
  FeatureDetector = cv2.xfeatures2d.SIFT_create(nOctaveLayers=8,contrastThreshold=0.04,edgeThreshold=10,sigma=2.0)
  FeatureMatcher  = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
  images = []
  kps = []
  dts = []
  size = len(files)
  for file in files:
    images.append(load_and_resize(file))
  idx = 0
  image = images[idx]
  ones = np.ones(tuple(DISPLAY_SIZE[::-1]))
  kp0,dt0 = FeatureDetector.detectAndCompute(image, None)
  kp1,dt1 = kp0,dt0
  T   = np.eye(3)
  T[0,2] = (CANVAS_WIDTH2 - image.shape[1]) // 2
  T[1,2] = (CANVAS_HEIGHT2 - image.shape[0]) // 2
  VideoCanvas2 = cv2.warpPerspective(image,T,(CANVAS_WIDTH2, CANVAS_HEIGHT2), flags=WARP_FLAGS).astype(np.float)
  CanvasCount2 = cv2.warpPerspective(ones,T,(CANVAS_WIDTH2, CANVAS_HEIGHT2), flags=WARP_FLAGS).astype(np.float)
  cv2.imshow(WINDOW_NAME, np.array(VideoCanvas2, dtype=np.uint8))
  cv2.waitKey(0)
  img0 = image
  img_mid = image
  sav_T   = None
  sav_img = None
  sav_kp  = None
  sav_dt  = None
  
  while idx < size - 1:
    idx += 1
    image = images[idx]
    print(files[idx])
    if idx == size - 1:
      T = sav_T
      img0 = sav_img
      kp1, dt1 = sav_kp, sav_dt
      
    kp2,dt2 = FeatureDetector.detectAndCompute(image, None)
    matches = sorted(FeatureMatcher.match(dt2, dt1), key=lambda x:x.distance)
    src, dst = [], []
    for m in matches:
      src.append(kp2[m.queryIdx].pt + (1,))
      dst.append(kp1[m.trainIdx].pt + (1,))
    src = np.array(src,dtype=np.float)
    dst = np.array(dst,dtype=np.float)
    A, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=1.0)
    T = T.dot(A)
    warp_img = cv2.warpPerspective(image,T,(CANVAS_WIDTH2, CANVAS_HEIGHT2), flags=WARP_FLAGS).astype(np.float)
    t_count  = cv2.warpPerspective(ones,T,(CANVAS_WIDTH2, CANVAS_HEIGHT2), flags=WARP_FLAGS).astype(np.float)
    if idx == 5:
      sav_T = T
      sav_dt = dt2
      sav_kp = kp2
      sav_img = image
    VideoCanvas2 += warp_img
    CanvasCount2 += t_count.astype(np.float)
    t_count = CanvasCount2.copy()
    t_count[t_count == 0] = 1
    disp = VideoCanvas2.copy() 
    disp[:,:,0] = VideoCanvas2[:,:,0] / t_count
    disp[:,:,1] = VideoCanvas2[:,:,1] / t_count
    disp[:,:,2] = VideoCanvas2[:,:,2] / t_count
    cv2.imshow(WINDOW_NAME, np.array(disp, dtype=np.uint8))
    cv2.imshow('matching',cv2.drawMatches(image,kp2,img0,kp1,matches[:CANDIDATE_NUM*5], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS))
    img0 = image
    kp1,dt1 = kp2,dt2
    key = cv2.waitKey(20)
    if key == 27:
      return
  key = cv2.waitKey(0)

def stitich_unordered(files, method):
  global VideoCanvas, CanvasCount
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
stitich_ordered_homography(files, VideoCanvas)
stitich_ordered(files[::-1], METHOD_STACKING, 0)

files = glob(f"dataset2/*.JPG")
stitich_unordered(files, METHOD_STACKING)
stitich_pizza(files)

cv2.destroyAllWindows()
