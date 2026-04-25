import os
import csv
import json
import time
import signal
import queue
import argparse
import threading
import multiprocessing as mp
from datetime import datetime
import cv2
import numpy as np
from utils import ArducamUtils


stop_event = threading.Event()
SUPPORTED_OUTPUT_FORMATS = {"h264_mkv", "h264_all_intra_mkv", "mjpeg_mkv"}


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


def build_recording_profile(cfg_recording):
    output_format = str(cfg_recording["output_format"]).strip().lower()
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise RuntimeError(f"Unsupported recording.output_format: {output_format}")

    if output_format == "mjpeg_mkv":
        jpeg_quality = int(cfg_recording["mjpeg"]["jpeg_quality"])
        if jpeg_quality < 1 or jpeg_quality > 100:
            raise RuntimeError(f"recording.mjpeg.jpeg_quality must be in [1, 100], got {jpeg_quality}")
        return {
            "output_format": output_format,
            "codec_name": "mjpeg",
            "encoder_name": "jpegenc",
            "encoder_threads": None,
            "bitrate_kbps": None,
            "speed_preset": None,
            "effective_gop": None,
            "effective_tune": None,
            "jpeg_quality": jpeg_quality,
        }

    h264_cfg = cfg_recording["h264"]
    encoder_name = h264_cfg["encoder_impl"].strip() if h264_cfg.get("encoder_impl") else "x264enc"
    effective_gop = 1 if output_format == "h264_all_intra_mkv" else int(h264_cfg["gop"])
    effective_tune = "zerolatency" if output_format == "h264_all_intra_mkv" else h264_cfg["tune"]
    return {
        "output_format": output_format,
        "codec_name": "h264",
        "encoder_name": encoder_name,
        "encoder_threads": int(max(1, h264_cfg["encoder_threads"])),
        "bitrate_kbps": int(h264_cfg["bitrate"]),
        "speed_preset": h264_cfg["speed_preset"],
        "effective_gop": effective_gop,
        "effective_tune": effective_tune,
        "jpeg_quality": None,
    }


def build_mkv_pipeline(
    output_path,
    width,
    height,
    fps,
    recording_profile,
    gst_queue_max_buffers,
):
    sink_path = output_path.replace("\\", "\\\\").replace('"', '\\"')
    queue_max_buffers = int(max(1, gst_queue_max_buffers))
    base = (
        "appsrc ! "
        f"video/x-raw,format=GRAY8,width={int(width)},height={int(height)},framerate={int(max(1, round(fps)))}/1 ! "
        f"queue max-size-buffers={queue_max_buffers} max-size-time=0 max-size-bytes=0 ! "
        "videoconvert ! "
    )

    if recording_profile["output_format"] == "mjpeg_mkv":
        return (
            base
            + f"jpegenc quality={int(recording_profile['jpeg_quality'])} ! "
            + "matroskamux ! "
            + f'filesink location="{sink_path}" sync=false'
        )

    return (
        base
        + f"{recording_profile['encoder_name']} bitrate={int(recording_profile['bitrate_kbps'])} "
        + f"key-int-max={int(recording_profile['effective_gop'])} bframes=0 "
        + "byte-stream=true "
        + f"speed-preset={recording_profile['speed_preset']} "
        + f"tune={recording_profile['effective_tune']} "
        + f"threads={int(recording_profile['encoder_threads'])} ! "
        "h264parse ! "
        "matroskamux ! "
        f'filesink location="{sink_path}" sync=false'
    )


def normalize_recording_config(cfg_recording):
    normalized = dict(cfg_recording)
    if "output_format" not in normalized:
        raise RuntimeError("recording.output_format is required")

    normalized["output_format"] = str(normalized["output_format"]).strip().lower()
    if normalized["output_format"] not in SUPPORTED_OUTPUT_FORMATS:
        raise RuntimeError(
            f"recording.output_format must be one of {sorted(SUPPORTED_OUTPUT_FORMATS)}, "
            f"got {normalized['output_format']}"
        )

    normalized["h264"] = dict(normalized.get("h264", {}))
    normalized["mjpeg"] = dict(normalized.get("mjpeg", {}))

    if normalized["output_format"] in {"h264_mkv", "h264_all_intra_mkv"}:
        required_h264 = ["encoder_impl", "encoder_threads", "bitrate", "gop", "speed_preset", "tune"]
        for key in required_h264:
            if key not in normalized["h264"]:
                raise RuntimeError(f"recording.h264.{key} is required when output_format={normalized['output_format']}")

        normalized["h264"]["encoder_threads"] = int(normalized["h264"]["encoder_threads"])
        normalized["h264"]["bitrate"] = int(normalized["h264"]["bitrate"])
        normalized["h264"]["gop"] = int(normalized["h264"]["gop"])

    if normalized["output_format"] == "mjpeg_mkv":
        if "jpeg_quality" not in normalized["mjpeg"]:
            raise RuntimeError("recording.mjpeg.jpeg_quality is required when output_format=mjpeg_mkv")
        normalized["mjpeg"]["jpeg_quality"] = int(normalized["mjpeg"]["jpeg_quality"])

    build_recording_profile(normalized)
    return normalized


def normalize_camera_configs(cfg):
    if "cameras" in cfg:
        cameras = cfg["cameras"]
    else:
        cameras = [cfg["camera"]]

    if not isinstance(cameras, list) or not cameras:
        raise RuntimeError("Config must contain a non-empty cameras list")
    if len(cameras) > 2:
        raise RuntimeError("This recorder supports at most 2 sync boards")

    normalized = []
    board_ids = set()
    for idx, camera in enumerate(cameras):
        if not bool(camera.get("enable", True)):
            continue

        board_id = str(camera.get("id", f"board{idx}"))
        if not board_id or "/" in board_id or "\\" in board_id:
            raise RuntimeError(f"Invalid camera id: {board_id}")
        if board_id in board_ids:
            raise RuntimeError(f"Duplicate camera id: {board_id}")
        board_ids.add(board_id)

        camera_cfg = dict(camera)
        if str(camera_cfg.get("pixelformat", "")).strip().upper() != "GREY":
            raise RuntimeError(f"{board_id} pixelformat must be GREY for gray passthrough recording")
        camera_cfg["id"] = board_id
        camera_cfg["enable"] = True
        normalized.append(camera_cfg)

    if not normalized:
        raise RuntimeError("At least one camera entry must be enabled")

    return normalized


def board_output_path(session_dir, board_id, cfg_recording):
    file_prefix = cfg_recording.get("file_prefix", "wide")
    return os.path.join(session_dir, f"{board_id}_{file_prefix}.mkv")


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


def ensure_gray8_frame(frame, context):
    if frame is None:
        raise RuntimeError(f"{context} returned no frame")
    if frame.ndim != 2:
        raise RuntimeError(f"{context} must be a single-channel GREY frame, got shape {frame.shape}")
    if frame.dtype != np.uint8:
        raise RuntimeError(f"{context} must be uint8 GREY, got dtype {frame.dtype}")
    return frame


class CameraSource:
    def __init__(self, cfg_camera):
        self.cfg_camera = cfg_camera
        self.board_id = cfg_camera["id"]
        self.device = cfg_camera["device"]
        self.arducam_utils = None
        self.cap = None
        self.raw_shape = None
        self.frame_size = None

    def open(self):
        self.arducam_utils = ArducamUtils(self.device)
        self.cap = open_camera(self.cfg_camera)
        self.arducam_utils.refresh()
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, self.arducam_utils.convert2rgb)

        channel = int(self.cfg_camera.get("channel", -1))
        if channel in range(0, 4):
            self.arducam_utils.write_dev(ArducamUtils.CHANNEL_SWITCH_REG, channel)

        ret, frame = self.cap.read()
        if not ret:
            self.release()
            raise RuntimeError(f"Failed to read first frame from {self.board_id}")

        if self.arducam_utils.convert2rgb == 0:
            raw_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            raw_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.raw_shape = (raw_height, raw_width)

        frame = convert_frame(frame, self.arducam_utils, self.raw_shape)
        frame = ensure_gray8_frame(frame, f"{self.board_id} first capture")
        self.frame_size = (frame.shape[1], frame.shape[0])
        return self.frame_size

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, None, None, None
        capture_time_wall = time.time()
        capture_time_mono_ns = time.monotonic_ns()
        frame = convert_frame(frame, self.arducam_utils, self.raw_shape)
        frame = ensure_gray8_frame(frame, f"{self.board_id} capture")
        return True, frame, capture_time_wall, capture_time_mono_ns

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.arducam_utils is not None:
            self.arducam_utils.close()
            self.arducam_utils = None


class WideStreamRecorder:
    def __init__(self, board_id, frame_size, cfg_recording, container_fps, output_path):
        self.board_id = board_id
        self.frame_size = frame_size
        self.container_fps = float(container_fps)
        self.recording_profile = build_recording_profile(cfg_recording)
        self.queue_size = int(max(1, cfg_recording["queue_size"]))
        self.gst_queue_max_buffers = int(max(1, cfg_recording.get("gst_queue_max_buffers", 8)))
        self.save_timestamps = bool(cfg_recording["save_timestamps"])

        self.output_path = output_path
        self.timestamps_path = self.output_path.rsplit(".", 1)[0] + "_timestamps.csv"
        self.summary_path = self.output_path.rsplit(".", 1)[0] + "_info.json"
        self.summary = None

        pipeline = build_mkv_pipeline(
            output_path=self.output_path,
            width=self.frame_size[0],
            height=self.frame_size[1],
            fps=self.container_fps,
            recording_profile=self.recording_profile,
            gst_queue_max_buffers=self.gst_queue_max_buffers,
        )

        self.writer = cv2.VideoWriter(
            pipeline,
            cv2.CAP_GSTREAMER,
            0,
            self.container_fps,
            self.frame_size,
            False,
        )

        self.frame_queue = queue.Queue(maxsize=self.queue_size)
        self.stop_write_event = threading.Event()
        self.writer_lock = threading.Lock()
        self.thread = None

        self.timestamps_file = None
        self.csv_writer = None

        self.received_frames = 0
        self.enqueued_frames = 0
        self.written_frames = 0
        self.dropped_frames = 0
        self.failed_enqueue_frames = 0
        self.writer_thread_alive = False

        self.start_time_wall = None
        self.end_time_wall = None
        self.start_time_mono = None

        self.first_capture_ts = None
        self.last_capture_ts = None
        self.first_capture_mono_ns = None
        self.last_capture_mono_ns = None
        self.first_capture_frame_id = None
        self.last_capture_frame_id = None
        self.first_written_capture_ts = None
        self.last_written_capture_ts = None
        self.first_written_capture_mono_ns = None
        self.last_written_capture_mono_ns = None
        self.first_written_capture_frame_id = None
        self.last_written_capture_frame_id = None

    def start(self):
        if not self.writer.isOpened():
            return False

        if self.save_timestamps:
            self.timestamps_file = open(self.timestamps_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.timestamps_file)
            self.csv_writer.writerow(
                ["frame_id", "capture_frame_id", "timestamp_unix", "timestamp_mono_ns", "relative_time_sec"]
            )
            self.timestamps_file.flush()

        self.start_time_wall = time.time()
        self.start_time_mono = time.monotonic_ns()

        self.thread = threading.Thread(target=self._run, name=f"{self.board_id}-writer", daemon=True)
        self.thread.start()
        return True

    def enqueue(self, frame, capture_frame_id, capture_time_wall, capture_time_mono_ns):
        ensure_gray8_frame(frame, f"{self.board_id} enqueue")
        frame_to_write = np.ascontiguousarray(frame)

        self.received_frames += 1
        if self.first_capture_ts is None:
            self.first_capture_ts = capture_time_wall
            self.first_capture_mono_ns = capture_time_mono_ns
            self.first_capture_frame_id = capture_frame_id
        self.last_capture_ts = capture_time_wall
        self.last_capture_mono_ns = capture_time_mono_ns
        self.last_capture_frame_id = capture_frame_id

        item = (frame_to_write, capture_frame_id, capture_time_wall, capture_time_mono_ns)
        self._enqueue_or_drop_oldest(item)

    def _enqueue_or_drop_oldest(self, item):
        try:
            self.frame_queue.put_nowait(item)
            self.enqueued_frames += 1
            return
        except queue.Full:
            pass

        try:
            self.frame_queue.get_nowait()
            self.dropped_frames += 1
            self.frame_queue.put_nowait(item)
            self.enqueued_frames += 1
        except queue.Empty:
            self.failed_enqueue_frames += 1
        except queue.Full:
            self.failed_enqueue_frames += 1

    def _run(self):
        while not self.stop_write_event.is_set() or not self.frame_queue.empty():
            try:
                frame_to_write, capture_frame_id, cap_wall, cap_mono_ns = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            with self.writer_lock:
                if not self.writer.isOpened():
                    continue

                self.writer.write(frame_to_write)
                frame_id = self.written_frames

                if self.first_written_capture_ts is None:
                    self.first_written_capture_ts = cap_wall
                    self.first_written_capture_mono_ns = cap_mono_ns
                    self.first_written_capture_frame_id = capture_frame_id
                self.last_written_capture_ts = cap_wall
                self.last_written_capture_mono_ns = cap_mono_ns
                self.last_written_capture_frame_id = capture_frame_id

                if self.save_timestamps and self.csv_writer is not None:
                    rel = (cap_mono_ns - self.start_time_mono) / 1e9 if self.start_time_mono else 0
                    self.csv_writer.writerow(
                        [frame_id, capture_frame_id, f"{cap_wall:.9f}", str(cap_mono_ns), f"{rel:.9f}"]
                    )
                    if frame_id % 120 == 0:
                        self.timestamps_file.flush()

                self.written_frames += 1

    def stop(self, timeout_sec, drain_queue=True):
        self.stop_write_event.set()
        if not drain_queue:
            self._discard_pending_frames()
        if self.thread is not None:
            self.thread.join(timeout=timeout_sec)
            self.writer_thread_alive = self.thread.is_alive()
            if self.writer_thread_alive:
                print(f"Warning: {self.board_id} writer thread did not stop within {timeout_sec:.2f}s.")

        self.end_time_wall = time.time()

        if not self.writer_thread_alive:
            with self.writer_lock:
                if self.writer.isOpened():
                    self.writer.release()
                if self.timestamps_file is not None:
                    self.timestamps_file.flush()
                    self.timestamps_file.close()
                    self.timestamps_file = None
                    self.csv_writer = None

        self.summary = self._write_summary()
        return self.summary

    def _discard_pending_frames(self):
        while True:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

    def _write_summary(self):
        capture_duration = 0
        actual_fps = 0
        if self.first_written_capture_ts is not None and self.last_written_capture_ts is not None:
            capture_duration = self.last_written_capture_ts - self.first_written_capture_ts
            if capture_duration > 0 and self.written_frames > 1:
                actual_fps = (self.written_frames - 1) / capture_duration

        summary = {
            "board_id": self.board_id,
            "video_file": os.path.basename(self.output_path),
            "timestamps_file": os.path.basename(self.timestamps_path) if self.save_timestamps else None,
            "frame_size": {"width": self.frame_size[0], "height": self.frame_size[1]},
            "container_fps": self.container_fps,
            "output_format": self.recording_profile["output_format"],
            "codec_name": self.recording_profile["codec_name"],
            "encoder_name": self.recording_profile["encoder_name"],
            "encoder_threads": self.recording_profile["encoder_threads"],
            "bitrate_kbps": self.recording_profile["bitrate_kbps"],
            "speed_preset": self.recording_profile["speed_preset"],
            "effective_gop": self.recording_profile["effective_gop"],
            "effective_tune": self.recording_profile["effective_tune"],
            "jpeg_quality": self.recording_profile["jpeg_quality"],
            "received_frames": self.received_frames,
            "enqueued_frames": self.enqueued_frames,
            "written_frames": self.written_frames,
            "dropped_frames": self.dropped_frames,
            "failed_enqueue_frames": self.failed_enqueue_frames,
            "queue_final_size": self.frame_queue.qsize(),
            "drop_rate_percent": (self.dropped_frames / self.received_frames * 100) if self.received_frames > 0 else 0,
            "start_time": self.start_time_wall,
            "end_time": self.end_time_wall,
            "first_capture_timestamp_unix": self.first_capture_ts,
            "last_capture_timestamp_unix": self.last_capture_ts,
            "first_capture_mono_ns": self.first_capture_mono_ns,
            "last_capture_mono_ns": self.last_capture_mono_ns,
            "first_capture_frame_id": self.first_capture_frame_id,
            "last_capture_frame_id": self.last_capture_frame_id,
            "first_written_capture_timestamp_unix": self.first_written_capture_ts,
            "last_written_capture_timestamp_unix": self.last_written_capture_ts,
            "first_written_capture_mono_ns": self.first_written_capture_mono_ns,
            "last_written_capture_mono_ns": self.last_written_capture_mono_ns,
            "first_written_capture_frame_id": self.first_written_capture_frame_id,
            "last_written_capture_frame_id": self.last_written_capture_frame_id,
            "capture_duration": capture_duration,
            "video_duration_by_metadata": self.written_frames / self.container_fps if self.container_fps > 0 else 0,
            "actual_fps_from_timestamps": actual_fps,
            "writer_thread_alive_at_stop": self.writer_thread_alive,
            "timing_note": "Frame-to-time alignment should use timestamps CSV/JSON metadata, not container FPS.",
        }

        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary


def put_status(status_queue, status):
    status["status_time"] = time.time()
    try:
        status_queue.put_nowait(status)
    except queue.Full:
        pass


def make_error_summary(cfg_camera, cfg_recording, session_dir, status, error_message, start_time, end_time):
    board_id = cfg_camera["id"]
    output_path = board_output_path(session_dir, board_id, cfg_recording)
    timestamps_path = output_path.rsplit(".", 1)[0] + "_timestamps.csv"
    recording_profile = build_recording_profile(cfg_recording)
    summary = {
        "board_id": board_id,
        "device": cfg_camera["device"],
        "video_file": os.path.basename(output_path),
        "timestamps_file": os.path.basename(timestamps_path) if bool(cfg_recording["save_timestamps"]) else None,
        "frame_size": None,
        "container_fps": float(cfg_recording["container_fps"]),
        "output_format": recording_profile["output_format"],
        "codec_name": recording_profile["codec_name"],
        "encoder_name": recording_profile["encoder_name"],
        "encoder_threads": recording_profile["encoder_threads"],
        "bitrate_kbps": recording_profile["bitrate_kbps"],
        "speed_preset": recording_profile["speed_preset"],
        "effective_gop": recording_profile["effective_gop"],
        "effective_tune": recording_profile["effective_tune"],
        "jpeg_quality": recording_profile["jpeg_quality"],
        "received_frames": 0,
        "enqueued_frames": 0,
        "written_frames": 0,
        "dropped_frames": 0,
        "failed_enqueue_frames": 0,
        "queue_final_size": 0,
        "drop_rate_percent": 0,
        "start_time": start_time,
        "end_time": end_time,
        "first_capture_timestamp_unix": None,
        "last_capture_timestamp_unix": None,
        "first_capture_mono_ns": None,
        "last_capture_mono_ns": None,
        "first_capture_frame_id": None,
        "last_capture_frame_id": None,
        "first_written_capture_timestamp_unix": None,
        "last_written_capture_timestamp_unix": None,
        "first_written_capture_mono_ns": None,
        "last_written_capture_mono_ns": None,
        "first_written_capture_frame_id": None,
        "last_written_capture_frame_id": None,
        "capture_duration": 0,
        "video_duration_by_metadata": 0,
        "actual_fps_from_timestamps": 0,
        "writer_thread_alive_at_stop": False,
        "captured_frames": 0,
        "read_failures": 0,
        "status": status,
        "error": error_message,
        "forced_stop": False,
    }

    summary_path = output_path.rsplit(".", 1)[0] + "_info.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def write_board_summary_file(session_dir, summary):
    info_path = os.path.join(session_dir, summary["video_file"].rsplit(".", 1)[0] + "_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def run_board_recorder_process(
    cfg_camera,
    cfg_runtime,
    cfg_recording,
    cfg_system,
    session_dir,
    stop_event_mp,
    capture_start_event,
    status_queue,
    result_queue,
):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    board_id = cfg_camera["id"]
    source = None
    recorder = None
    captured_frames = 0
    read_failures = 0
    child_status = "completed"
    error_message = None
    process_start_time = time.time()
    process_end_time = None
    smooth_fps = 0.0
    fps_alpha = 0.12
    last_fps_t = time.perf_counter()
    last_status_t = 0.0

    put_status(status_queue, {"type": "status", "board_id": board_id, "status": "starting"})

    try:
        source = CameraSource(cfg_camera)
        frame_size = source.open()
        output_path = board_output_path(session_dir, board_id, cfg_recording)
        container_fps = float(cfg_recording["container_fps"])
        recorder = WideStreamRecorder(
            board_id=board_id,
            frame_size=frame_size,
            cfg_recording=cfg_recording,
            container_fps=container_fps,
            output_path=output_path,
        )
        if not recorder.start():
            raise RuntimeError(
                f"Failed to create {cfg_recording['output_format']} writer for {board_id} via GStreamer. "
                "Check OpenCV GStreamer support and required encoder plugins."
            )

        print(f"{board_id}: /dev/video{source.device} -> {recorder.output_path}")
        if recorder.save_timestamps:
            print(f"{board_id}: timestamps -> {recorder.timestamps_path}")
        print(f"{board_id}: frame size {frame_size[0]}x{frame_size[1]}")
        put_status(
            status_queue,
            {
                "type": "status",
                "board_id": board_id,
                "status": "ready",
                "frame_width": frame_size[0],
                "frame_height": frame_size[1],
            },
        )
        while not stop_event_mp.is_set() and not capture_start_event.is_set():
            time.sleep(0.01)

        show_display = bool(cfg_runtime.get("show_display", False))
        headless_sleep = float(cfg_runtime.get("sleep_sec_headless", 0.0005))
        max_fps = cfg_runtime.get("max_fps")
        max_fps = float(max_fps) if max_fps is not None else None
        frame_interval = 1.0 / max_fps if max_fps else 0
        next_frame_time = time.perf_counter()

        if show_display:
            cv2.namedWindow(f"{board_id} Wide Preview", cv2.WINDOW_NORMAL)

        while not stop_event_mp.is_set():
            now = time.perf_counter()
            if frame_interval > 0 and now < next_frame_time:
                continue
            if frame_interval > 0:
                next_frame_time = now + frame_interval

            ret, frame, ts_wall, ts_mono_ns = source.read()
            if not ret:
                read_failures += 1
                child_status = "error"
                error_message = f"Failed to read frame from {board_id}"
                stop_event_mp.set()
                break

            capture_frame_id = captured_frames
            recorder.enqueue(frame, capture_frame_id, ts_wall, ts_mono_ns)
            captured_frames += 1

            t = time.perf_counter()
            dt = t - last_fps_t
            if dt > 0:
                inst = 1.0 / dt
                smooth_fps = fps_alpha * inst + (1.0 - fps_alpha) * smooth_fps
            last_fps_t = t

            if show_display:
                cv2.imshow(f"{board_id} Wide Preview", frame)
                key = cv2.waitKey(1)
                if key == ord("q") or key == 27:
                    stop_event_mp.set()
                    break
            else:
                time.sleep(headless_sleep)

            if t - last_status_t >= 1.0:
                last_status_t = t
                put_status(
                    status_queue,
                    {
                        "type": "status",
                        "board_id": board_id,
                        "status": "recording",
                        "captured_frames": captured_frames,
                        "written_frames": recorder.written_frames,
                        "queue_size": recorder.frame_queue.qsize(),
                        "dropped_frames": recorder.dropped_frames,
                        "fps": smooth_fps,
                    },
                )

        if child_status == "completed" and stop_event_mp.is_set():
            child_status = "stopped"
    except Exception as exc:
        child_status = "error"
        error_message = f"{type(exc).__name__}: {exc}"
        stop_event_mp.set()
    finally:
        process_end_time = time.time()
        if recorder is not None:
            parent_timeout_sec = float(cfg_system["graceful_shutdown_timeout_sec"])
            fast_stop = child_status == "error" and read_failures > 0
            summary = recorder.stop(
                timeout_sec=1.0 if fast_stop else max(1.0, parent_timeout_sec * 0.5),
                drain_queue=not fast_stop,
            )
            if summary.get("writer_thread_alive_at_stop") and child_status != "error":
                child_status = "error"
                error_message = "Writer thread did not stop cleanly"
            summary.update(
                {
                    "device": cfg_camera["device"],
                    "captured_frames": captured_frames,
                    "read_failures": read_failures,
                    "process_start_time": process_start_time,
                    "process_end_time": process_end_time,
                    "status": child_status,
                    "error": error_message,
                    "forced_stop": False,
                }
            )
            with open(recorder.summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        else:
            summary = make_error_summary(
                cfg_camera=cfg_camera,
                cfg_recording=cfg_recording,
                session_dir=session_dir,
                status=child_status,
                error_message=error_message,
                start_time=process_start_time,
                end_time=process_end_time,
            )

        if source is not None:
            source.release()
        if bool(cfg_runtime.get("show_display", False)):
            cv2.destroyAllWindows()

        put_status(status_queue, {"type": "status", "board_id": board_id, "status": child_status, "error": error_message})
        result_queue.put(summary)


class RecordingSession:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cameras = normalize_camera_configs(cfg)
        self.cfg_runtime = cfg["runtime"]
        self.cfg_recording = normalize_recording_config(cfg["recording"])
        self.cfg_system = cfg["system"]
        self.timeout_sec = float(self.cfg_system["graceful_shutdown_timeout_sec"])

        self.session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        output_dir = self.cfg_recording["output_dir"]
        self.session_dir = os.path.join(output_dir, self.session_id)
        self.session_info_path = os.path.join(self.session_dir, "session_info.json")

        self.ctx = mp.get_context("spawn")
        self.process_stop_event = self.ctx.Event()
        self.capture_start_event = self.ctx.Event()
        self.status_queue = self.ctx.Queue(maxsize=max(16, len(self.cameras) * 8))
        self.result_queue = self.ctx.Queue()
        self.processes = []
        self.latest_status = {}
        self.board_summaries = {}
        self.start_time_wall = None
        self.end_time_wall = None
        self.capture_released_time = None

    def start(self):
        os.makedirs(self.session_dir, exist_ok=True)
        for cfg_camera in self.cameras:
            self._start_board_process(cfg_camera)
            self._wait_for_board_ready(cfg_camera)

        self.start_time_wall = time.time()
        self.capture_released_time = self.start_time_wall
        self.capture_start_event.set()

    def _start_board_process(self, cfg_camera):
        process = self.ctx.Process(
            target=run_board_recorder_process,
            name=f"{cfg_camera['id']}-recorder",
            args=(
                cfg_camera,
                self.cfg_runtime,
                self.cfg_recording,
                self.cfg_system,
                self.session_dir,
                self.process_stop_event,
                self.capture_start_event,
                self.status_queue,
                self.result_queue,
            ),
        )
        process.start()
        self.processes.append({"camera": cfg_camera, "process": process, "forced_stop": False})

    def _wait_for_board_ready(self, cfg_camera):
        board_id = cfg_camera["id"]
        timeout_sec = float(self.cfg_system["board_startup_timeout_sec"])
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            self._drain_status()
            self._drain_results()

            summary = self.board_summaries.get(board_id)
            if summary is not None and summary.get("status") == "error":
                self.process_stop_event.set()
                raise RuntimeError(f"{board_id} failed during startup: {summary.get('error')}")

            status = self.latest_status.get(board_id)
            if status is not None and status.get("status") == "ready":
                delay_sec = float(self.cfg_system["board_startup_delay_sec"])
                if delay_sec > 0:
                    time.sleep(delay_sec)
                return

            process = self.processes[-1]["process"]
            if not process.is_alive():
                self.process_stop_event.set()
                raise RuntimeError(f"{board_id} recorder process exited during startup")

            time.sleep(0.05)

        self.process_stop_event.set()
        raise RuntimeError(f"{board_id} did not become ready within {timeout_sec:.2f}s")

    def wait(self):
        fps_log = bool(self.cfg_runtime.get("fps_log", True))
        while not stop_event.is_set():
            self._drain_status()
            self._drain_results()

            if self._has_failed_board():
                stop_event.set()
                break
            if self._has_process_without_summary():
                stop_event.set()
                break
            if self._mark_stale_boards():
                stop_event.set()
                break
            if self._all_processes_done():
                break

            if fps_log:
                print(self._status_line(), end="\r")
            time.sleep(1.0)

    def stop(self):
        self.process_stop_event.set()
        for item in self.processes:
            process = item["process"]
            join_timeout = 0.5 if item.get("force_now") else self.timeout_sec
            process.join(timeout=join_timeout)
            if process.is_alive():
                item["forced_stop"] = True
                process.terminate()
                process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)

        self._drain_status()
        self._drain_results()

        board_summaries = []
        for item in self.processes:
            cfg_camera = item["camera"]
            board_id = cfg_camera["id"]
            summary = self.board_summaries.get(board_id)
            if summary is None:
                status = "forced_stop" if item["forced_stop"] else "error"
                error = "Recorder process did not return a summary"
                summary = make_error_summary(
                    cfg_camera=cfg_camera,
                    cfg_recording=self.cfg_recording,
                    session_dir=self.session_dir,
                    status=status,
                    error_message=error,
                    start_time=self.start_time_wall,
                    end_time=time.time(),
                )

            process = item["process"]
            summary["forced_stop"] = bool(item["forced_stop"])
            summary["process_exitcode"] = process.exitcode
            if item["forced_stop"] and not summary.get("error"):
                status = self.latest_status.get(board_id, {})
                summary["error"] = status.get("error", "Recorder process was forced to stop")
            write_board_summary_file(self.session_dir, summary)
            board_summaries.append(summary)

        self.end_time_wall = time.time()
        session_summary = self._build_session_summary(board_summaries)
        with open(self.session_info_path, "w", encoding="utf-8") as f:
            json.dump(session_summary, f, indent=2)
        return session_summary

    def _drain_status(self):
        while True:
            try:
                item = self.status_queue.get_nowait()
            except queue.Empty:
                break
            if item.get("type") == "status":
                self.latest_status[item["board_id"]] = item

    def _drain_results(self):
        while True:
            try:
                item = self.result_queue.get_nowait()
            except queue.Empty:
                break
            self.board_summaries[item["board_id"]] = item

    def _has_failed_board(self):
        for summary in self.board_summaries.values():
            if summary.get("status") == "error" or summary.get("forced_stop"):
                return True
        return False

    def _all_processes_done(self):
        return all(not item["process"].is_alive() for item in self.processes)

    def _has_process_without_summary(self):
        for item in self.processes:
            process = item["process"]
            board_id = item["camera"]["id"]
            if not process.is_alive() and process.exitcode is not None and board_id not in self.board_summaries:
                return True
        return False

    def _mark_stale_boards(self):
        if self.capture_released_time is None:
            return False
        timeout_sec = float(self.cfg_system["board_status_timeout_sec"])
        now = time.time()
        for item in self.processes:
            process = item["process"]
            board_id = item["camera"]["id"]
            status = self.latest_status.get(board_id)
            if status is None or not process.is_alive():
                continue
            if status.get("status") not in ("ready", "recording"):
                continue
            last_status_time = max(float(status.get("status_time", now)), self.capture_released_time)
            if now - last_status_time <= timeout_sec:
                continue
            item["force_now"] = True
            status["status"] = "stale"
            status["error"] = f"No frames/status from {board_id} for {timeout_sec:.2f}s"
            return True
        return False

    def _status_line(self):
        parts = []
        for item in self.processes:
            board_id = item["camera"]["id"]
            status = self.latest_status.get(board_id, {"status": "starting"})
            process = item["process"]
            parts.append(
                f"{board_id}: {status.get('status')} | "
                f"FPS {float(status.get('fps', 0)):.2f} | "
                f"Captured {int(status.get('captured_frames', 0))} | "
                f"Written {int(status.get('written_frames', 0))} | "
                f"Q {int(status.get('queue_size', 0))} | "
                f"Drop {int(status.get('dropped_frames', 0))} | "
                f"PID {process.pid}"
            )
        return " || ".join(parts)

    def _build_session_summary(self, board_summaries):
        first_mono_values = [
            item["first_capture_mono_ns"]
            for item in board_summaries
            if item.get("first_capture_mono_ns") is not None
        ]
        first_mono_delta_ns = None
        if len(first_mono_values) >= 2:
            first_mono_delta_ns = max(first_mono_values) - min(first_mono_values)

        failed_boards = [
            item["board_id"]
            for item in board_summaries
            if item.get("status") == "error" or item.get("forced_stop")
        ]

        return {
            "session_id": self.session_id,
            "session_dir": self.session_dir,
            "start_time": self.start_time_wall,
            "end_time": self.end_time_wall,
            "board_count": len(board_summaries),
            "failed_board_count": len(failed_boards),
            "failed_boards": failed_boards,
            "first_capture_mono_delta_ns": first_mono_delta_ns,
            "recording": dict(self.cfg_recording),
            "process_model": "one_process_per_sync_board",
            "boards": board_summaries,
            "timing_note": "Two-board alignment is based on host-side capture timestamps, not hardware exposure timestamps.",
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Wide capture recorder (configurable MKV encoders)")
    parser.add_argument(
        "-c",
        "--config",
        default="cam_cap_config.json",
        help="Path to config JSON file (default: cam_cap_config.json)",
    )
    return parser.parse_args()


def print_session_summary(session, session_summary):
    print("=" * 70)
    print("WIDE CAPTURE SESSION SUMMARY")
    print("=" * 70)
    for board in session_summary["boards"]:
        print(f"{board['board_id']}:")
        print(f"  Status: {board['status']}")
        print(f"  Format: {board['output_format']}")
        print(f"  Captured frames: {board['captured_frames']}")
        print(f"  Written frames: {board['written_frames']}")
        print(f"  Real FPS(from written timestamps): {board['actual_fps_from_timestamps']:.2f}")
        print(f"  Dropped(write queue): {board['dropped_frames']}")
        print(f"  Forced stop: {board['forced_stop']}")
        if board.get("error"):
            print(f"  Error: {board['error']}")
        print(f"  Video: {os.path.join(session.session_dir, board['video_file'])}")
        if board["timestamps_file"]:
            print(f"  Timestamps: {os.path.join(session.session_dir, board['timestamps_file'])}")
        print(f"  Summary JSON: {os.path.join(session.session_dir, board['video_file'].rsplit('.', 1)[0] + '_info.json')}")
    print(f"Session JSON: {session.session_info_path}")


def run():
    args = parse_args()
    cfg = load_config(args.config)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    session = RecordingSession(cfg)
    session_summary = None
    try:
        session.start()
        print(f"Session: {session.session_dir}")
        print(f"Container FPS(metadata only): {float(session.cfg_recording['container_fps']):.2f}")
        session.wait()
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        print("\n\nShutting down...")
        session_summary = session.stop()
        print_session_summary(session, session_summary)

    return 1 if session_summary["failed_board_count"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(run())
