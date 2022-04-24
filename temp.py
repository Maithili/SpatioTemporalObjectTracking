import glob
import os
import shutil

dir = 'logs_fiveclusters0419'
for train_step_dir in os.listdir(dir):
    images = glob.glob(dir+'/'+train_step_dir+'/all/*.jpg')
    print(images)
    steps = train_step_dir[-3:]
    for image in images:
        target_file = image.replace(train_step_dir+'/all', 'comparison').replace('.jpg',steps+'.jpg')
        shutil.copyfile(image, target_file)