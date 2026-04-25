## Camera Setup
Install driver:
```bash
wget https://github.com/ArduCAM/MIPI_Camera/releases/download/v0.0.3/install_full.sh
chmod +x install_full.sh
./install_full.sh -m arducam
```

Build Docker image:
```bash
docker build -t cam:dev .
```

Run with Docker:
```bash
sudo docker run --rm -it --privileged --device=/dev/video0 -v $PWD:/work -w /work cam:dev python3 recorder/cam_cap.py --config cam_cap_config.json
```

Alternatively, you can also directly run:
```bash
python cam_cap.py --config cam_cap_config.json
```

Split a recorded session into per-camera streams:

```bash
python split_wide_video.py your/recordings/path/session_YYYYmmdd_HHMMSS
```

Dependencies:
- OpenCV
- v4l2
- Numpy

## Notes
The frame rate in MKV containers has no practical meaning, frame metadata should be referenced from the CSV file.

## cam_cap.json Configuration Options
```json
{
  "cameras": [
    {
      "id": "board0",
      "enable": true,
      "device": 0,  # /dev device
      "pixelformat": "GREY",
      "width": 5120,
      "height": 800,
      "channel": -1,
      "capture_buffersize": 8
    },
    {
      "id": "board1",
      "enable": true,
      "device": 1,  # /dev device
      "pixelformat": "GREY",
      "width": 5120,
      "height": 800,
      "channel": -1,
      "capture_buffersize": 8
    }
  ],
  "runtime": {
    "show_display": false,  # Show live preview window while recording.
    "fps_log": true,    # Print real-time FPS and queue/drop status in terminal.
    "max_fps": null,    # Optional software capture-rate limit. Set to null to let the gray recording path run as fast as it can.
    "sleep_sec_headless": 0.0005    # Small sleep interval in headless mode to reduce CPU busy-wait.
  },
  "recording": {
    "output_dir": "recordings",
    "file_prefix": "wide",
    "output_format": "mjpeg_mkv",   # Required MKV output mode. Supported values are h264_mkv, h264_all_intra_mkv, and mjpeg_mkv.
    "gst_queue_max_buffers": 32,
    "queue_size": 64,
    "save_timestamps": true,    
    "container_fps": 30,    # FPS metadata written into video container; timing alignment should rely on timestamp metadata.
    "h264": {   # Nested config used by h264_mkv and h264_all_intra_mkv
      "encoder_impl": "x264enc",
      "encoder_threads": 4,
      "bitrate": 12000,
      "gop": 30,
      "speed_preset": "ultrafast",
      "tune": "zerolatency"
    },
    "mjpeg": {  #  Nested config used by mjpeg_mkv
      "jpeg_quality": 85
    }
  },
  "system": {
    "graceful_shutdown_timeout_sec": 20.0,
    "board_startup_timeout_sec": 30.0,
    "board_startup_delay_sec": 2.0,
    "board_status_timeout_sec": 5.0
  }
}

```

## Output Layout

Each run creates one session directory:

```text
recordings/session_YYYYmmdd_HHMMSS/
  session_info.json
  board0_wide.mkv
  board0_wide_timestamps.csv
  board0_wide_info.json
  board1_wide.mkv
  board1_wide_timestamps.csv
  board1_wide_info.json
```

The recorder saves one wide video per sync board. Per-camera streams are generated after recording by the offline splitter.

Timestamp CSV rows use this schema:

```text
frame_id,capture_frame_id,timestamp_unix,timestamp_mono_ns,relative_time_sec
```

`frame_id` is the contiguous frame index actually written to the MKV. `capture_frame_id` is the
contiguous index assigned immediately after a successful `cap.read()` in that board process, before
reshape, validation, enqueue, or encode. `timestamp_unix` and `timestamp_mono_ns` are captured at
that same point. Use `timestamp_mono_ns` first for cross-board alignment and use `capture_frame_id`
gaps to locate capture frames that were later dropped by the writer queue.

Board summary JSON and `session_info.json` record the actual `output_format` and effective encoder
settings used by that session.

## Offline Split

Split a recorded session into per-camera streams:

```bash
python recorder/split_wide_video.py recordings/session_YYYYmmdd_HHMMSS
```

The default output directory is:

```text
recordings/session_YYYYmmdd_HHMMSS/split_streams/
  cam0.mkv
  cam0_timestamps.csv
  cam0_info.json
  ...
  split_info.json
```

For an 8-camera session, `board0` maps to `cam0` through `cam3`, and `board1` maps to `cam4` through `cam7`.
The splitter writes per-camera MKV files as single-channel gray streams and copies each board's
timestamp CSV unchanged. By default it re-encodes split streams using the same `output_format` and
`jpeg_quality` recorded in the source session metadata.
