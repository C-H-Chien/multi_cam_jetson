# sudo python3 ./arducam_pair_displayer.py -f GREY -d0 0 -d1 1 --fps

try:
    import cv2
except ImportError:
    print("Start to install opencv...")
    os.system(f'sudo apt-get update')
    os.system(f'sudo apt install nvidia-opencv-dev')
import numpy as np
from datetime import datetime
import array
import fcntl
import os
import argparse
import keyboard
try:
    from utils import ArducamUtils
except ImportError as e:
    import sys

    print(e)
    print("Start to install python environment...")

    if sys.version[0] == 2:
        print("Try to install python-pip...")
        os.system(f'sudo apt install python-pip')       
    if sys.version[0] == 3:
        print("Try to install python3-pip...")
        os.system(f'sudo apt install python3-pip')

    try:
        from pip import main as pipmain
    except ImportError:
        from pip._internal import main as pipmain

    print("Try to install jetson-stats...")
    pipmain(['install', 'jetson-stats'])

    print("Try to install v4l2-fix...")
    pipmain(['install', 'v4l2-fix'])

    from utils import ArducamUtils
import time
import sys

def resize(frame, dst_width):
    width = frame.shape[1]
    height = frame.shape[0]
    scale = dst_width * 1.0 / width
    return cv2.resize(frame, (int(scale * width), int(scale * height)))

def display(cap_0, cap_1, arducam_utils_0, arducam_utils_1, fps = False, cap_imgs = False):
    counter = 0
    current_frame = 0;
    start_time = datetime.now()
    frame_count = 0
    start = time.time()
    try:
    	while True:    			
    		ret_0, frame_0 = cap_0.read()
    		ret_1, frame_1 = cap_1.read()
    		counter += 1
    		frame_count += 1

    		if arducam_utils_0.convert2rgb == 0:
        	    w_0 = cap_0.get(cv2.CAP_PROP_FRAME_WIDTH)
        	    h_0 = cap_0.get(cv2.CAP_PROP_FRAME_HEIGHT)
        	    frame_0 = frame_0.reshape(int(h_0), int(w_0))
    		if arducam_utils_1.convert2rgb == 0:
        	    w_1 = cap_1.get(cv2.CAP_PROP_FRAME_WIDTH)
        	    h_1 = cap_1.get(cv2.CAP_PROP_FRAME_HEIGHT)
        	    frame_1 = frame_1.reshape(int(h_1), int(w_1))

    		frame_0 = arducam_utils_0.convert(frame_0)
    		frame_1 = arducam_utils_1.convert(frame_1)
        
    		frame_0 = resize(frame_0, 1280.0)
    		frame_1 = resize(frame_1, 1280.0)
        	
        	# display
        	#cv2.imshow("Arducam_0", frame_0)
        	#cv2.imshow("Arducam_1", frame_1)
    		stack_vert_frame = np.concatenate((frame_0, frame_1), axis=0)
    		cv2.imshow("Arducam", stack_vert_frame)
    		cv2.waitKey(1)
        	
    		if keyboard.is_pressed('g'):
        	    img0_shot_name = "/home/jetsonlems/Arducam_Imgs/" + str(current_frame) + "_0.png"
        	    cv2.imwrite(img0_shot_name, stack_vert_frame)
        	    #img1_shot_name = "/home/jetsonlems/Arducam_Imgs/" + str(current_frame) + "_1.png"
#        	    print("Taking screenshot of two quadroscopic cameras ...")
#        	    cv2.imwrite(img0_shot_name, frame_0);
#        	    cv2.imwrite(img1_shot_name, frame_1);
        	    current_frame += 1
        	    
    		if fps and time.time() - start >= 1:
        	    if sys.version[0] == '2':
        	        print("fps: {}".format(frame_count))    
        	    else:
        	        print("fps: {}".format(frame_count),end='\r')
        	    start = time.time()
        	    frame_count = 0 
    except KeyboardInterrupt:
    	end_time = datetime.now()
    	elapsed_time = end_time - start_time
    	avgtime = elapsed_time.total_seconds() / counter
    	print ("Average time between frames: " + str(avgtime))
    	print ("Average FPS: " + str(1/avgtime))


def fourcc(a, b, c, d):
    return ord(a) | (ord(b) << 8) | (ord(c) << 16) | (ord(d) << 24)

def pixelformat(string):
    if len(string) != 3 and len(string) != 4:
        msg = "{} is not a pixel format".format(string)
        raise argparse.ArgumentTypeError(msg)
    if len(string) == 3:
        return fourcc(string[0], string[1], string[2], ' ')
    else:
        return fourcc(string[0], string[1], string[2], string[3])

def show_info(arducam_utils):
    _, firmware_version = arducam_utils.read_dev(ArducamUtils.FIRMWARE_VERSION_REG)
    _, sensor_id = arducam_utils.read_dev(ArducamUtils.FIRMWARE_SENSOR_ID_REG)
    _, serial_number = arducam_utils.read_dev(ArducamUtils.SERIAL_NUMBER_REG)
    print("Firmware Version: {}".format(firmware_version))
    print("Sensor ID: 0x{:04X}".format(sensor_id))
    print("Serial Number: 0x{:08X}".format(serial_number))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arducam Jetson Nano MIPI Camera Displayer.')

    parser.add_argument('-d0', '--device0', default=0, type=int, nargs='?',
                        help='/dev/videoX default is 0')
    parser.add_argument('-d1', '--device1', default=1, type=int, nargs='?',
                        help='/dev/videoX default is 1')
    parser.add_argument('-f', '--pixelformat', type=pixelformat,
                        help="set pixelformat")
    parser.add_argument('--width', type=lambda x: int(x,0),
                        help="set width of image")
    parser.add_argument('--height', type=lambda x: int(x,0),
                        help="set height of image")
    parser.add_argument('--fps', action='store_true', help="display fps")
    parser.add_argument('--cap', action='store_true', help="capture images")
    parser.add_argument('--channel', type=int, default=-1, nargs='?',
                        help="When using Camarray's single channel, use this parameter to switch channels. \
                            (E.g. ov9781/ov9281 Quadrascopic Camera Bundle Kit)")

    args = parser.parse_args()

    # open camera
    cap_0 = cv2.VideoCapture(args.device0, cv2.CAP_V4L2)
    cap_1 = cv2.VideoCapture(args.device1, cv2.CAP_V4L2)

    # set pixel format
    if args.pixelformat != None:
        if not cap_0.set(cv2.CAP_PROP_FOURCC, args.pixelformat):
            print("Failed to set pixel format for cap_0.")
        if not cap_1.set(cv2.CAP_PROP_FOURCC, args.pixelformat):
            print("Failed to set pixel format for cap_1.")

    arducam_utils_0 = ArducamUtils(args.device0)
    arducam_utils_1 = ArducamUtils(args.device1)

    show_info(arducam_utils_0)
    show_info(arducam_utils_1)
    
    print(arducam_utils_0.depth)
    
    # turn off RGB conversion
    if arducam_utils_0.convert2rgb == 0:
        cap_0.set(cv2.CAP_PROP_CONVERT_RGB, arducam_utils_0.convert2rgb)
    if arducam_utils_1.convert2rgb == 0:
        cap_1.set(cv2.CAP_PROP_CONVERT_RGB, arducam_utils_1.convert2rgb)
    # set width
    if args.width != None:
        cap_0.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap_1.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    # set height
    if args.height != None:
        cap_0.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap_1.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if args.channel in range(0, 4):
        arducam_utils_0.write_dev(ArducamUtils.CHANNEL_SWITCH_REG, args.channel)
        arducam_utils_1.write_dev(ArducamUtils.CHANNEL_SWITCH_REG, args.channel)

    # begin display
    display(cap_0, cap_1, arducam_utils_0, arducam_utils_1, args.fps)

    # release camera
    cap_0.release()
    cap_1.release()
