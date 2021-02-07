# %%
import os
import random as r
import shutil

data_path = r'..\Real Data\hv18\full'
bb_path = r'..\Real Data\hv18\bounding box'
file_path = []
for (root, dirs, files) in os.walk(data_path):
    for file_name in files:
        path = os.path.join(root,file_name)
        file_path.append(path)

r.shuffle(file_path)

for i in range(len(file_path)):
    bb_path = r'..\Real Data\hv18\bounding box'

    im_path = file_path[i]
    list_form = im_path.split(os.path.sep)
    file_name = list_form[-1]
    label = list_form[-2]

    bb_dest_list = bb_path.split(os.path.sep)
    bb_dest_list[-1] = label
    bb_dest_list.append(file_name)
    bb_dest = os.path.sep.join(bb_dest_list)

    bb_path_list = bb_path.split(os.path.sep)
    bb_path_list.append(file_name)
    bb_or = os.path.sep.join(bb_path_list)

    try:
        shutil.move(bb_or, bb_dest)
    except(FileNotFoundError):
        print(file_name)
        print(label)



