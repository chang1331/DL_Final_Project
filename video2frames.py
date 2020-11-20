import cv2
import os
import imutils

try:
    current_file_dir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
except NameError:
    current_file_dir = os.getcwd().replace('\\', '/')

video820Path = current_file_dir + '/820_1.mp4'
vs = cv2.VideoCapture(video820Path)

try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))
    # an error occurred while trying to determine the total
    # number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

frameIndex = 0
while True:
    (grabbed, frame) = vs.read()
    if not grabbed:
        break
    # if frameIndex < 8000:
    #    frameIndex += 1
    #    continue
    cv2.imwrite(current_file_dir + '/820_1' + '/frame_{}.jpeg'.format(frameIndex), frame)
    frameIndex += 1

print("[INFO] cleaning up...")
vs.release()
