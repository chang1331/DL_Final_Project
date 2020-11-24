import os
import pandas as pd
import numpy as np

try:
    current_file_dir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
except NameError:
    current_file_dir = os.getcwd().replace('\\', '/')

# dir of individual frame labels
input_label_dir = 'C:/Users/Administrator/Desktop/BBox-Label-Tool-master/Labels/001/'
label_file_list = os.listdir(input_label_dir)

# dir of frames
input_frame_dir = current_file_dir + '/data/images/'
frame_file_list = os.listdir(input_frame_dir)

assert len(label_file_list) == len(frame_file_list)

final_list = []
for frame in frame_file_list:
    assert '.jpeg' in frame, 'wrong image type'
    frame_name = frame[:frame.find('.')]
    assert frame_name + '.txt' in label_file_list

    label_df = pd.read_csv(input_label_dir + frame_name + '.txt', delim_whitespace=True, skiprows=1, header=None,
                           names=['x1', 'y1', 'x2', 'y2'])
    if label_df.empty:
        final_list.append([0, 0, 0, 0, frame_name, 0])  # x1,y1,x2,y2,frame_name,label
    else:
        label_df['frame_name'] = frame_name
        label_df['label'] = 1
        final_list = final_list + label_df.values.tolist()

np.savetxt(current_file_dir + '/data/labels.csv', final_list, delimiter=",", fmt='% s')

