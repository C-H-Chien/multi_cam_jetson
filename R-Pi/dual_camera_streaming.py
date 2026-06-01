#!/usr/bin/env python3

import csv
import time
import cv2
import os
from picamera2 import Picamera2

os.makedirs("stacked", exist_ok=True)

#> Get CAM0 and CAM1
cam0 = Picamera2(0)
cam1 = Picamera2(1)

config0 = cam0.create_video_configuration( main={"size": (5120, 800), "format": "YUV420"} )
config1 = cam1.create_video_configuration( main={"size": (5120, 800), "format": "YUV420"} )

cam0.configure(config0)
cam1.configure(config1)

cam0.start()
cam1.start()

time.sleep(2)
b_write_stacked_images = 0

#> Timestamp log
csv_file = open("timestamps.csv", "w", newline="")
writer = csv.writer(csv_file)

writer.writerow([
    "frame_id",
    "cam0_timestamp_ns",
    "cam1_timestamp_ns",
    "delta_us"
])

frame_id = 0

try:
    while True:

        #> Capture latest frame and metadata
        frame0 = cam0.capture_array("main")
        meta0 = cam0.capture_metadata()

        frame1 = cam1.capture_array("main")
        meta1 = cam1.capture_metadata()

        ts0 = meta0.get("SensorTimestamp", -1)
        ts1 = meta1.get("SensorTimestamp", -1)
        
        if b_write_stacked_images:
            stacked_frame = cv2.vconcat([frame0, frame1])        
            filename = (f"stacked/frame_{frame_id:06d}_{ts0}_{ts1}.png")
            cv2.imwrite(filename, stacked_frame)

        delta_us = (ts1 - ts0) / 1000.0
        writer.writerow([ frame_id, ts0, ts1, delta_us ])

        print(f"Frame {frame_id:06d} CAM0={ts0} CAM1={ts1} Δ={delta_us:.1f} us")

        #> Display
        #> Resize the original 5120x800 image per quadrascopic bundle kit to fit the monitor screen size
        disp0 = cv2.resize(frame0, (1280, 200))
        disp1 = cv2.resize(frame1, (1280, 200))

        cv2.imshow("CAM0", disp0)
        cv2.imshow("CAM1", disp1)

        key = cv2.waitKey(1)

        if key == ord('q'):
            break

        frame_id += 1

except KeyboardInterrupt:
    pass

finally:
    csv_file.close()

    cam0.stop()
    cam1.stop()

    cv2.destroyAllWindows()
