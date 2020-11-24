import cv2
import imutils
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-videoName", default=None)
    args = parser.parse_args()

    try:
        current_file_dir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
    except NameError:
        current_file_dir = os.getcwd().replace('\\', '/')

    print('[INFO] Current file directory: {}'.format(current_file_dir))

    assert 'videos' in os.listdir(current_file_dir), 'there is no folder named videos in current file directory'
    assert args.videoName is not None, 'missing argument videoName'

    video_name = str(args.videoName)  # e.g., oct2_920

    assert '{}.mp4'.format(video_name) in os.listdir(
        current_file_dir + '/videos/'), 'there is no {}.mp4 under folder videos'.format(video_name)
    assert video_name in os.listdir(current_file_dir), 'there is no {} folder under current file directory'.format(
        video_name)

    vs = cv2.VideoCapture(current_file_dir + '/videos/{}.mp4'.format(video_name))

    # count the total number of frames in the video
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] {} total frames in video".format(total))
    except:
        print("[INFO] Cannot determine # of frames in video")
        total = -1

    # extract video frames
    idx = 0
    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break
        if idx % 40 == 0:
            cv2.imwrite(current_file_dir + '/{}/{}_frame_{}.jpeg'.format(video_name, video_name, idx), frame)
            idx += 1
        else:
            idx += 1
            continue

    print("[INFO] Cleaning up...")
    vs.release()
    print('[INFO] Finished.')


if __name__ == '__main__':
    main()
