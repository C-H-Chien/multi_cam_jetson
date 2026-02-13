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
sudo docker run --rm -it --privileged --device=/dev/video0 -v $PWD:/work -w /work cam:dev python3 $python_file_name$
```

Run:
```bash
python cam_cap.py --config cam_cap_config.json
```

packages:
` opencv v4l2-fix numpy `


## cam_cap.json Configuration Options

### camera

- **device**: V4L2 camera index. `0` means `/dev/video0`.
- **pixelformat**: Input FourCC format string (3 or 4 chars), for example `GREY`.
- **width**: Requested capture width in pixels.
- **height**: Requested capture height in pixels.
- **channel**: Optional camera channel switch index for multi-channel adapters. Use `-1` to disable channel switching.
- **capture_buffersize**: OpenCV/V4L2 capture buffer count. Lower values reduce latency.

### runtime

- **show_display**: Show live preview window while recording.
- **fps_log**: Print real-time FPS and queue/drop status in terminal.
- **max_fps**: Optional software capture-rate limit. Use `null` to disable limiting.
- **sleep_sec_headless**: Small sleep interval in headless mode to reduce CPU busy-wait.

### recording

- **output_dir**: Directory for output files.
- **file_prefix**: Prefix of generated output files.
- **encoder_impl**: GStreamer encoder element name. Current default is `x264enc` (software encoding).
- **gst_queue_max_buffers**: Max buffered frames in internal GStreamer queue. Use a small value (for example `4~16`) to prevent memory growth when encoder is slower than capture.
- **queue_size**: Max frame queue size before dropping oldest frames.
- **save_timestamps**: Whether to write per-frame timestamp CSV metadata.
- **container_fps**: FPS metadata written into video container; timing alignment should rely on timestamp metadata.
- **bitrate**: Encoder target bitrate in kbps for `x264enc`.
- **gop**: Keyframe interval (Group of Pictures) in frames.
- **speed_preset**: x264 speed preset (quality/CPU tradeoff), for example `ultrafast`.
- **tune**: x264 tune profile, default `zerolatency`.

### system

- **graceful_shutdown_timeout_sec**: Max seconds to wait for writer thread flush during shutdown.

