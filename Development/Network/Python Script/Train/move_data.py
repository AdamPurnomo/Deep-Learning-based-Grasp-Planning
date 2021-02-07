# %%
import os
import random as r
import shutil

data_path = r'..\..\..\Simulation\Image\Training Data\hv18\Positive\full'
file_path = []
for (root, dirs, files) in os.walk(data_path):
    for file_name in files:
        path = os.path.join(root,file_name)
        file_path.append(path)

r.shuffle(file_path)

for i in range(15700):
    im_path = file_path[i]
    list_form = im_path.split(os.path.sep)
    list_form[-2] = 'bounding box'
    bb_path = os.path.sep.join(list_form)
    
    im_dest = im_path.split(os.path.sep)
    im_dest[-2] = 'temp full'
    im_dest = os.path.sep.join(im_dest)
    shutil.move(im_path, im_dest)

    bb_dest = bb_path.split(os.path.sep)
    bb_dest[-2] = 'temp bounding box'
    bb_dest = os.path.sep.join(bb_dest)
    shutil.move(bb_path, bb_dest)



