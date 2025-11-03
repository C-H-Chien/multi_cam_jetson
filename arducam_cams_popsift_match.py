# sudo python3 ./arducam_cams_popsift_match.py -f GREY -d 0 --fps
import numpy as np
from datetime import datetime
import array
import fcntl
import os
import argparse
import cv2
import sys
from tqdm import trange

from utils import ArducamUtils
import time

# Try to import the PopSift modules
try:
    #from popsift_pybind import popsift_config
    #from popsift_pybind import popsift_extract
    #from popsift_pybind import popsift_match
    from popsift import popsift_match
    POPSIFT_AVAILABLE = True
except ImportError as e:
    print(f"Error: Could not import PopSift modules: {e}")
    print("Make sure the modules are compiled and in your Python path.")
    POPSIFT_AVAILABLE = False

cam_img_width = 1280

def match_features_from_arrays(left_array, right_array, print_timing=False, sift_config=None, _verbose=False):
	"""Match SIFT features between two numpy arrays."""

	if sift_config is not None:
	    result = popsift_match.match_features_from_arrays_with_config(
		left_array, right_array, sift_config,
		verbose=_verbose,
		print_time_info=print_timing
	    )
	else:
	    result = popsift_match.match_features_from_arrays(
		left_array, right_array,
		verbose=_verbose,
		print_time_info=print_timing
	    )

	if _verbose:
	    print(f"Found {result.num_matches} matches")
	    if print_timing:
	    	print(f"Matching time: {result.match_time_ms:.2f} ms")

	return result

def resize(frame, dst_width):
    width = frame.shape[1]
    height = frame.shape[0]
    scale = dst_width * 1.0 / width
    return cv2.resize(frame, (int(scale * width), int(scale * height)))

def get_sift_match(cap, arducam_utils, fps = False):
    #counter = 0
    start_time = datetime.now()
    frame_count = 0
    
    time_init_stack = []
    time_enqueue_stack = []
    time_processing_stack = []
    time_cleanup_stack = []
    time_total_GPU_extract_keyf_stack = []
    time_total_GPU_extract_curf_stack = []
    time_total_GPU_matching_stack = []
    time_total_batch_stack = []
    max_num_of_loops = 500
    
    start = time.time()
    try:
    	for counter in trange(max_num_of_loops):
        	ret, frame = cap.read()
        	frame_count += 1

        	frame = arducam_utils.convert(frame)
        	#frame = resize(frame, 2560.0)
        	if frame is not None and counter == 0:
        		print(frame.shape)
        		frame_h, frame_w, _ = frame.shape
        		cam_img_width = frame_w // 4
        		print(f"Frame size = ({frame_h}, {frame_w}), cam_img_width = {cam_img_width}")
        	
        	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        	cam1_img = frame[:, :cam_img_width]
        	cam2_img = frame[:, cam_img_width:cam_img_width*2]
        	cam3_img = frame[:, cam_img_width*2:cam_img_width*3]
        	cam4_img = frame[:, cam_img_width*3:]
        	
        	if counter == 0:
        		#keyf_images = [cam1_img, cam2_img, cam3_img, cam4_img, cam1_img, cam2_img, cam3_img, cam4_img]
        		keyf_images = [cam1_img, cam2_img, cam3_img, cam4_img]
        		continue
        	
        	#curr_images = [cam1_img, cam2_img, cam3_img, cam4_img, cam1_img, cam2_img, cam3_img, cam4_img]
        	curr_images = [cam1_img, cam2_img, cam3_img, cam4_img]
        	
        	'''
        	#start = time.time()
        	results_single = []
        	for keyf, curf in zip(keyf_images, curr_images):
        		result = popsift_match.match_features_from_arrays(keyf, curf, verbose=False)
        		results_single.append(result)
        		
        	#time_single = time.time() - start
        	#print(f"Single processing time:  {time_single:.3f} seconds")
        	
        	# sum up all the timings 
        	time_init = sum(r.init_time_ms for r in results_single)
        	time_enqueue = sum(r.enqueue_time_ms for r in results_single)
        	time_cleanup = sum(r.cleanup_time_ms for r in results_single)
        	time_total_GPU_extract_keyf = sum(r.left_gpu_time_ms for r in results_single)
        	time_total_GPU_extract_curf = sum(r.right_gpu_time_ms for r in results_single)
        	time_total_GPU_matching = sum(r.match_time_ms for r in results_single)
        	'''
        	
        	
        	#start = time.time()
        	batch_results = popsift_match.match_multiple_pairs_from_arrays(
        		keyf_images,
         		curr_images,
        		verbose=False,
        		print_time_info=False
        	)
        	#time_batch = time.time() - start
        	
        	time_init = batch_results[0].init_time_ms
        	time_enqueue = batch_results[0].enqueue_time_ms
        	time_processing = batch_results[0].processing_time_ms
        	time_cleanup = batch_results[0].cleanup_time_ms
        	time_total_GPU_extract_keyf = sum(r.left_gpu_time_ms for r in batch_results)
        	time_total_GPU_extract_curf = sum(r.right_gpu_time_ms for r in batch_results)
        	time_total_GPU_extract = time_total_GPU_extract_keyf + time_total_GPU_extract_curf
        	time_total_GPU_matching = sum(r.match_time_ms for r in batch_results)
        	time_batch = batch_results[0].total_batch_time_ms
        	
        	
        	'''
        	print("Time breakdown:")
        	print(f"- Initialization: {time_init:.3f} ms")
        	print(f"- Enqueue images: {time_enqueue:.3f} ms")
        	print(f"- Processing (CPU+GPU): {time_processing:.3f} ms")
        	print(f"  - Batch total extraction time (GPU): {time_total_GPU_extract:.3f} ms")
        	print(f"  - Batch total matching time (GPU): {time_total_GPU_matching:.3f} ms")
        	print(f"- Cleanup: {time_cleanup:.3f} ms")
        	print(f"- Total processing time (CPU+GPU):  {time_batch:.3f} ms")
        	'''
        	
        	time_init_stack.append(time_init)
        	time_enqueue_stack.append(time_enqueue)
        	time_processing_stack.append(time_processing)
        	time_cleanup_stack.append(time_cleanup)
        	time_total_GPU_extract_keyf_stack.append(time_total_GPU_extract_keyf)
        	time_total_GPU_extract_curf_stack.append(time_total_GPU_extract_curf)
        	time_total_GPU_matching_stack.append(time_total_GPU_matching)
        	time_total_batch_stack.append(time_batch)
        	
        	'''
        	ret = cv2.waitKey(1)

        	if fps and time.time() - start >= 1:
        	    if sys.version[0] == '2':
        	        print("fps: {}".format(frame_count))    
        	    else:
        	        print("fps: {}".format(frame_count),end='\r')
        	    start = time.time()
        	    frame_count = 0 
        	'''
    except KeyboardInterrupt:
    	end_time = datetime.now()
    	elapsed_time = end_time - start_time
    	avgtime = elapsed_time.total_seconds() / counter
    	print ("Average time (s) between frames: " + str(avgtime))
    	print ("Average FPS: " + str(1/avgtime))
    
    write_num_precision = 10
    with open("/home/jetsonlems/cchien3/arducam_4_pair.txt", "w") as f:
    	for i in range(max_num_of_loops-1):
    		f.write(f"{time_init_stack[i]:.{write_num_precision}f}\t")
    		f.write(f"{time_enqueue_stack[i]:.{write_num_precision}f}\t")
    		f.write(f"{time_processing_stack[i]:.{write_num_precision}f}\t")
    		f.write(f"{time_cleanup_stack[i]:.{write_num_precision}f}\t")
    		f.write(f"{time_total_GPU_extract_keyf_stack[i]:.{write_num_precision}f}\t")
    		f.write(f"{time_total_GPU_extract_curf_stack[i]:.{write_num_precision}f}\t")
    		f.write(f"{time_total_GPU_matching_stack[i]:.{write_num_precision}f}\t")
    		f.write(f"{time_total_batch_stack[i]:.{write_num_precision}f}\n")


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

    parser.add_argument('-d', '--device', default=0, type=int, nargs='?',
                        help='/dev/videoX default is 0')
    parser.add_argument('-f', '--pixelformat', type=pixelformat,
                        help="set pixelformat")
    parser.add_argument('--width', type=lambda x: int(x,0),
                        help="set width of image")
    parser.add_argument('--height', type=lambda x: int(x,0),
                        help="set height of image")
    parser.add_argument('--fps', action='store_true', help="display fps")
    parser.add_argument('--channel', type=int, default=-1, nargs='?',
                        help="When using Camarray's single channel, use this parameter to switch channels. \
                            (E.g. ov9781/ov9281 Quadrascopic Camera Bundle Kit)")

    args = parser.parse_args()

    # open camera
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)

    # set pixel format
    if args.pixelformat != None:
        if not cap.set(cv2.CAP_PROP_FOURCC, args.pixelformat):
            print("Failed to set pixel format.")

    arducam_utils = ArducamUtils(args.device)

    show_info(arducam_utils)
    # turn off RGB conversion
    if arducam_utils.convert2rgb == 0:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, arducam_utils.convert2rgb)
    # set width
    if args.width != None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    # set height
    if args.height != None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if args.channel in range(0, 4):
        arducam_utils.write_dev(ArducamUtils.CHANNEL_SWITCH_REG, args.channel)

    # begin display
    get_sift_match(cap, arducam_utils, args.fps)

    # release camera
    cap.release()
