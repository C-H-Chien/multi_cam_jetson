import os
import csv
import json
import time
import signal
import queue
import argparse
import threading
from datetime import datetime
import cv2
import numpy as np
from utils import ArducamUtils


stop_event = threading.Event()


def signal_handler(signum, frame):
    print("\n\nReceived shutdown signal. Stopping gracefully...")
    stop_event.set()


def load_config(config_path):
    if not os.path.exists(config_path):
        raise RuntimeError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def pixelformat(string):
    if len(string) not in (3, 4):
        raise ValueError(f"{string} is not a pixel format")
    if len(string) == 3:
        string = f"{string} "
    return cv2.VideoWriter_fourcc(*string)


def build_h264_mkv_pipeline(
    output_path,
    width,
    height,
    fps,
    bitrate_kbps,
    gop,
    speed_preset,
    tune,
    encoder_impl,
    gst_queue_max_buffers,
):
    encoder = encoder_impl.strip() if encoder_impl else "x264enc"
    sink_path = output_path.replace('\\', '\\\\').replace('"', '\\"')
    queue_max_buffers = int(max(1, gst_queue_max_buffers))
    return (
        "appsrc ! "
        f"video/x-raw,format=BGR,width={int(width)},height={int(height)},framerate={int(max(1, round(fps)))}/1 ! "
        f"queue max-size-buffers={queue_max_buffers} max-size-time=0 max-size-bytes=0 leaky=downstream ! "
        "videoconvert ! "
        f"{encoder} bitrate={int(bitrate_kbps)} key-int-max={int(gop)} bframes=0 "
        f"byte-stream=true speed-preset={speed_preset} tune={tune} threads=0 ! "
        "h264parse ! "
        "matroskamux ! "
        f"filesink location=\"{sink_path}\" sync=false"
    )


class WideStreamRecorder:
    def __init__(self, frame_size, cfg_recording, container_fps):
        self.frame_size = frame_size
        self.container_fps = float(container_fps)
        self.queue_size = int(max(1, cfg_recording["queue_size"]))
        self.gst_queue_max_buffers = int(max(1, cfg_recording.get("gst_queue_max_buffers", 8)))
        self.save_timestamps = bool(cfg_recording["save_timestamps"])

        output_dir = cfg_recording["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = cfg_recording["file_prefix"]
        self.output_path = os.path.join(output_dir, f"{prefix}_{ts}.mkv")
        self.timestamps_path = self.output_path.rsplit(".", 1)[0] + "_timestamps.csv"
        self.summary_path = self.output_path.rsplit(".", 1)[0] + "_info.json"

        pipeline = build_h264_mkv_pipeline(
            output_path=self.output_path,
            width=self.frame_size[0],
            height=self.frame_size[1],
            fps=self.container_fps,
            bitrate_kbps=cfg_recording["bitrate"],
            gop=cfg_recording["gop"],
            speed_preset=cfg_recording["speed_preset"],
            tune=cfg_recording["tune"],
            encoder_impl=cfg_recording["encoder_impl"],
            gst_queue_max_buffers=self.gst_queue_max_buffers,
        )

        self.writer = cv2.VideoWriter(
            pipeline,
            cv2.CAP_GSTREAMER,
            0,
            self.container_fps,
            self.frame_size,
            True,
        )

        self.frame_queue = queue.Queue(maxsize=self.queue_size)
        self.stop_write_event = threading.Event()
        self.writer_lock = threading.Lock()
        self.thread = None

        self.timestamps_file = None
        self.csv_writer = None

        self.enqueued_frames = 0
        self.written_frames = 0
        self.dropped_frames = 0

        self.start_time_wall = None
        self.end_time_wall = None
        self.start_time_mono = None

        self.first_capture_ts = None
        self.last_capture_ts = None

    def start(self):
        if not self.writer.isOpened():
            return False

        if self.save_timestamps:
            self.timestamps_file = open(self.timestamps_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.timestamps_file)
            self.csv_writer.writerow(["frame_id", "timestamp_unix", "timestamp_mono_ns", "relative_time_sec"])
            self.timestamps_file.flush()

        self.start_time_wall = time.time()
        self.start_time_mono = time.monotonic_ns()

        self.thread = threading.Thread(target=self._run, name="wide-writer", daemon=True)
        self.thread.start()
        return True

    def enqueue(self, frame, capture_time_wall, capture_time_mono_ns):
        frame_to_write = np.ascontiguousarray(frame)
        if frame_to_write.ndim == 2:
            frame_to_write = cv2.cvtColor(frame_to_write, cv2.COLOR_GRAY2BGR)

        if self.first_capture_ts is None:
            self.first_capture_ts = capture_time_wall
        self.last_capture_ts = capture_time_wall

        item = (frame_to_write, capture_time_wall, capture_time_mono_ns)
        self._enqueue_or_drop_oldest(item)

    def _enqueue_or_drop_oldest(self, item):
        try:
            self.frame_queue.put_nowait(item)
            self.enqueued_frames += 1
            return
        except queue.Full:
            self.dropped_frames += 1

        try:
            _ = self.frame_queue.get_nowait()
            self.frame_queue.put_nowait(item)
            self.enqueued_frames += 1
        except (queue.Empty, queue.Full):
            return

    def _run(self):
        while not self.stop_write_event.is_set() or not self.frame_queue.empty():
            try:
                frame_to_write, cap_wall, cap_mono_ns = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            with self.writer_lock:
                if not self.writer.isOpened():
                    continue

                self.writer.write(frame_to_write)
                frame_id = self.written_frames

                if self.save_timestamps and self.csv_writer is not None:
                    rel = (cap_mono_ns - self.start_time_mono) / 1e9 if self.start_time_mono else 0
                    self.csv_writer.writerow([frame_id, f"{cap_wall:.9f}", str(cap_mono_ns), f"{rel:.9f}"])
                    if frame_id % 120 == 0:
                        self.timestamps_file.flush()

                self.written_frames += 1

    def stop(self, timeout_sec):
        self.stop_write_event.set()
        if self.thread is not None:
            self.thread.join(timeout=timeout_sec)
            if self.thread.is_alive():
                print(f"Warning: writer thread did not stop within {timeout_sec:.2f}s; forcing resource shutdown.")

        self.end_time_wall = time.time()

        with self.writer_lock:
            if self.writer.isOpened():
                self.writer.release()
            if self.timestamps_file is not None:
                self.timestamps_file.flush()
                self.timestamps_file.close()
                self.timestamps_file = None
                self.csv_writer = None

        self._write_summary()

    def _write_summary(self):
        capture_duration = 0
        actual_fps = 0
        if self.first_capture_ts is not None and self.last_capture_ts is not None:
            capture_duration = self.last_capture_ts - self.first_capture_ts
            if capture_duration > 0 and self.written_frames > 1:
                actual_fps = (self.written_frames - 1) / capture_duration

        attempted = self.enqueued_frames + self.dropped_frames
        summary = {
            "video_file": os.path.basename(self.output_path),
            "timestamps_file": os.path.basename(self.timestamps_path) if self.save_timestamps else None,
            "frame_size": {"width": self.frame_size[0], "height": self.frame_size[1]},
            "container_fps": self.container_fps,
            "enqueued_frames": self.enqueued_frames,
            "written_frames": self.written_frames,
            "dropped_frames": self.dropped_frames,
            "drop_rate_percent": (self.dropped_frames / attempted * 100) if attempted > 0 else 0,
            "start_time": self.start_time_wall,
            "end_time": self.end_time_wall,
            "capture_duration": capture_duration,
            "video_duration_by_metadata": self.written_frames / self.container_fps if self.container_fps > 0 else 0,
            "actual_fps_from_timestamps": actual_fps,
            "timing_note": "Frame-to-time alignment should use timestamps CSV/JSON metadata, not container FPS.",
        }

        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


def open_camera(cfg_camera):
    cap = cv2.VideoCapture(cfg_camera["device"], cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open /dev/video{cfg_camera['device']}")

    if cfg_camera.get("capture_buffersize") is not None:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, int(cfg_camera["capture_buffersize"]))

    pf = cfg_camera.get("pixelformat")
    if pf:
        cap.set(cv2.CAP_PROP_FOURCC, pixelformat(pf))

    if cfg_camera.get("width"):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg_camera["width"]))
    if cfg_camera.get("height"):
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg_camera["height"]))

    return cap


def convert_frame(raw_frame, arducam_utils, raw_shape):
    if raw_shape is not None:
        raw_frame = raw_frame.reshape(raw_shape)
    return arducam_utils.convert(raw_frame)


def parse_args():
    parser = argparse.ArgumentParser(description="Wide capture recorder (soft H264 + MKV)")
    parser.add_argument(
        "-c",
        "--config",
        default="cam_cap_config.json",
        help="Path to config JSON file (default: cam_cap_config.json)",
    )
    return parser.parse_args()


def run():
    args = parse_args()
    config_path = args.config
    cfg = load_config(config_path)
    cfg_camera = cfg["camera"]
    cfg_runtime = cfg["runtime"]
    cfg_rec = cfg["recording"]
    cfg_system = cfg["system"]
    timeout_sec = float(cfg_system["graceful_shutdown_timeout_sec"])

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    arducam_utils = ArducamUtils(cfg_camera["device"])

    cap = open_camera(cfg_camera)

    cap.set(cv2.CAP_PROP_CONVERT_RGB, arducam_utils.convert2rgb)

    channel = int(cfg_camera.get("channel", -1))
    if channel in range(0, 4):
        arducam_utils.write_dev(ArducamUtils.CHANNEL_SWITCH_REG, channel)

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Failed to read first frame")

    raw_shape = None
    if arducam_utils.convert2rgb == 0:
        raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_shape = (raw_height, raw_width)

    frame = convert_frame(frame, arducam_utils, raw_shape)

    frame_size = (frame.shape[1], frame.shape[0])

    container_fps = float(cfg_rec["container_fps"])

    recorder = WideStreamRecorder(frame_size=frame_size, cfg_recording=cfg_rec, container_fps=container_fps)
    if not recorder.start():
        cap.release()
        raise RuntimeError(
            "Failed to create H264+MKV writer via GStreamer. "
            "Check OpenCV GStreamer support and x264enc plugin availability."
        )

    print(f"Recording to: {recorder.output_path}")
    if recorder.save_timestamps:
        print(f"Timestamps: {recorder.timestamps_path}")
    print(f"Frame size: {frame_size[0]}x{frame_size[1]}")
    print(f"Container FPS(metadata only): {float(container_fps):.2f}")

    show_display = bool(cfg_runtime.get("show_display", False))
    fps_log = bool(cfg_runtime.get("fps_log", True))
    headless_sleep = float(cfg_runtime.get("sleep_sec_headless", 0.0005))

    max_fps = cfg_runtime.get("max_fps")
    max_fps = float(max_fps) if max_fps is not None else None
    frame_interval = 1.0 / max_fps if max_fps else 0
    next_frame_time = time.perf_counter()

    fps_alpha = 0.12
    smooth_fps = 0.0
    last_fps_t = time.perf_counter()

    captured_frames = 0

    if show_display:
        cv2.namedWindow("Wide Preview", cv2.WINDOW_NORMAL)

    try:
        while not stop_event.is_set():
            now = time.perf_counter()
            if frame_interval > 0 and now < next_frame_time:
                continue
            if frame_interval > 0:
                next_frame_time = now + frame_interval

            ret, frame = cap.read()
            if not ret:
                print("\nFailed to read frame")
                break

            frame = convert_frame(frame, arducam_utils, raw_shape)

            ts_wall = time.time()
            ts_mono_ns = time.monotonic_ns()
            recorder.enqueue(frame, ts_wall, ts_mono_ns)
            captured_frames += 1

            if show_display:
                cv2.imshow("Wide Preview", frame)
                key = cv2.waitKey(1)
                if key == ord("q") or key == 27:
                    break
            else:
                time.sleep(headless_sleep)

            if fps_log:
                t = time.perf_counter()
                dt = t - last_fps_t
                if dt > 0:
                    inst = 1.0 / dt
                    smooth_fps = fps_alpha * inst + (1.0 - fps_alpha) * smooth_fps
                    print(
                        f"FPS: {smooth_fps:.2f} | Captured: {captured_frames} | "
                        f"Q: {recorder.frame_queue.qsize()} | "
                        f"Drop(write): {recorder.dropped_frames}",
                        end="\r",
                    )
                last_fps_t = t
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        print("\n\nShutting down...")
        recorder.stop(timeout_sec=timeout_sec)
        cap.release()
        if show_display:
            cv2.destroyAllWindows()

        real_fps = 0.0
        if recorder.first_capture_ts is not None and recorder.last_capture_ts is not None:
            capture_duration = recorder.last_capture_ts - recorder.first_capture_ts
            if capture_duration > 0 and recorder.written_frames > 1:
                real_fps = (recorder.written_frames - 1) / capture_duration
        
        print("=" * 70)
        print("WIDE CAPTURE SUMMARY")
        print("=" * 70)
        print(f"Captured frames: {captured_frames}")
        print(f"Written frames: {recorder.written_frames}")
        print(f"Real FPS(from timestamps): {real_fps:.2f}")
        print(f"Dropped(write): {recorder.dropped_frames}")
        print(f"Video: {recorder.output_path}")
        if recorder.save_timestamps:
            print(f"Timestamps: {recorder.timestamps_path}")
        print(f"Summary JSON: {recorder.summary_path}")


if __name__ == "__main__":
    run()
