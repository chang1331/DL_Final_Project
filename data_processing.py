import os
import pandas as pd
import numpy as np
import shutil

try:
    current_file_dir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
except NameError:
    current_file_dir = os.getcwd().replace('\\', '/')

# get images from individual folders and put them into one folder called training_images
data_dir = current_file_dir + '/data/'
images_folder_list = list(filter(lambda f: 'images_' in f, os.listdir(data_dir)))
training_images_dir = data_dir + 'training_images'
for folder in images_folder_list:
    source = data_dir + folder
    for image in os.listdir(source):
        dummy = shutil.copy(source + '/{}'.format(image), training_images_dir)
frame_file_list = os.listdir(training_images_dir)

# get all frame label txt files and put them into one list
label_folder_list = list(filter(lambda f: 'labels' in f, os.listdir(data_dir)))
label_file_list = []
for folder in label_folder_list:
    label_file_list = label_file_list + os.listdir(data_dir + folder)

assert len(label_file_list) == len(frame_file_list), 'number of frames must equal number of labels!!!'

final_list = []
for frame in frame_file_list:
    assert '.jpeg' in frame, 'wrong image type'
    frame_name = frame[:frame.find('.')]
    assert frame_name + '.txt' in label_file_list

    for folder in label_folder_list:
        try:
            label_df = pd.read_csv(data_dir + folder + '/{}.txt'.format(frame_name), delim_whitespace=True,
                                   skiprows=1, header=None, names=['x1', 'y1', 'x2', 'y2'])
        except FileNotFoundError:
            continue
        else:
            break

    if label_df.empty:
        final_list.append([0, 0, 0, 0, frame_name, 0])  # x1,y1,x2,y2,frame_name,label
    else:
        label_df['frame_name'] = frame_name
        label_df['label'] = 1
        final_list = final_list + label_df.values.tolist()

np.savetxt(data_dir + 'training_labels.csv', final_list, delimiter=",", fmt='% s')
