# A picture taking script that records pictures from a video camera and saves them to disk.
# Arguments: 
#   * imgdir: the folder to save images to. Defaults to Training Data.
#   * resolution: the picture resolution in NxY format. Default is 1920x1080.
#   * label: The label that applies to all of the images that will be taken. 
#            If not provided on command line, the script will promt for it before continuing.
# python3 picture_taker.py --imgdir=Sparrow --resolution=1920x1080
# The program will ask for a label, all pictures will be assigned this label.
# Adapted from https://github.com/EdjeElectronics/Image-Dataset-Tools/blob/main/PictureTaker/PictureTaker.py by Evan Juras, EJ Technology Consultants

import cv2
import os
import argparse
import sys
import shortuuid
from datetime import datetime

CAMERA_NUM = 1


# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--imgdir', help='Folder to save images in (will be created if it doesn\'t exist already',
                   default='training_data')
parser.add_argument('--label', help='The label of the pictures that will be taken. A folder will be created with this name if it does not already exist.')
parser.add_argument('--resolution', help='Desired camera resolution in WxH.',
                   default='1920x1080')
args = parser.parse_args()

labelname = args.label
if labelname is None:
    labelname = input("Enter the label of the pictures that will be taken:")

dirname = args.imgdir
if not 'x' in args.resolution:
    print('Please specify resolution as WxH. (example: 1920x1080)')
    sys.exit()
imW = int(args.resolution.split('x')[0])
imH = int(args.resolution.split('x')[1])
print("resolution extracted from {}: width={}, hight={}".format(args.resolution, imW, imH))

# Create output directory if it doesn't already exist
cwd = os.getcwd()
dirpath = os.path.join(cwd,dirname)
if not os.path.exists(dirpath):
    os.makedirs(dirpath)

# Create label directory if it doesn't already exist
subdirpath = os.path.join(dirpath, labelname)
cwd = os.getcwd()
#dirpath = os.path.join(cwd,subdirpath)
if not os.path.exists(subdirpath):
    os.makedirs(subdirpath)

# Creating file name template
date = datetime.today().strftime('%Y%m%d')    
example_filename = labelname + "_" + date + "_" + shortuuid.uuid() + ".jpg" 
savepath = os.path.join(subdirpath, example_filename)
print("Example filename: {}".format(savepath))

# Initialize webcam
cap = cv2.VideoCapture(CAMERA_NUM)
ret = cap.set(3, imW)
ret = cap.set(4, imH)

# Initialize display window
winname = 'Press \"p\" or spacebar to take a picture.'
cv2.namedWindow(winname)
cv2.moveWindow(winname,50,30)

print('Press "p" or spacebar to take a picture. Pictures will automatically be saved in the %s folder.' % subdirpath)
print('Press "q" to quit.')

while True:
    hasFrame, frame = cap.read()
    frame = cv2.flip(frame, 1) 
    frame = cv2.flip(frame, 0) 
    cv2.imshow(winname,frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    # 32 is space key
    elif key == ord('p') or key == 32: 
        #Take a picture!
        filename = labelname + "_" + date + "_" + shortuuid.uuid() + ".jpg" 
        savepath = os.path.join(subdirpath, filename)
        cv2.imwrite(savepath, frame)
        print('Picture taken and saved as %s' % filename)

cv2.destroyAllWindows()
cap.release()