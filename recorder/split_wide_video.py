#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys

import cv2


RECORDER_DIR = os.path.abspath(os.path.dirname(__file__))
if RECORDER_DIR not in sys.path:
    sys.path.insert(0, RECORDER_DIR)

from cam_cap import build_mkv_pipeline, build_recording_profile, normalize_recording_config


def ensure_gray_frame(frame, context):
    if frame is None:
        raise RuntimeError(f"{context} returned no frame")
    if len(frame.shape) == 2:
        return frame
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raise RuntimeError(f"{context} must decode to gray or BGR frame, got shape {frame.shape}")


def load_session_info(session_dir):
    path = os.path.join(session_dir, "session_info.json")
    if not os.path.exists(path):
        raise RuntimeError(f"session_info.json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def progress_bar(label, current, total):
    if total <= 0:
        if current % 100 == 0:
            print(f"{label}: {current} frames", end="\r")
        return

    width = 32
    filled = int(width * current / total)
    bar = "#" * filled + "." * (width - filled)
    percent = current / total * 100
    print(f"{label}: [{bar}] {percent:6.2f}% ({current}/{total})", end="\r")


def copy_timestamps(session_dir, board_summary, output_path):
    timestamps_file = board_summary.get("timestamps_file")
    if not timestamps_file:
        return None

    source_path = os.path.join(session_dir, timestamps_file)
    target_path = output_path.rsplit(".", 1)[0] + "_timestamps.csv"
    shutil.copyfile(source_path, target_path)
    return os.path.basename(target_path)


def make_writer(output_path, frame_size, fps, cfg_recording):
    pipeline = build_mkv_pipeline(
        output_path=output_path,
        width=frame_size[0],
        height=frame_size[1],
        fps=fps,
        recording_profile=build_recording_profile(cfg_recording),
        gst_queue_max_buffers=cfg_recording.get("gst_queue_max_buffers", 8),
    )
    return cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, fps, frame_size, False)


def split_board(session_dir, output_dir, board_summary, board_index, cfg_recording, slots_per_board):
    video_path = os.path.join(session_dir, board_summary["video_file"])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open wide video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = float(board_summary["container_fps"])

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Failed to read first frame from {video_path}")

    frame = ensure_gray_frame(frame, video_path)
    frame_height, frame_width = frame.shape[:2]
    if frame_width % slots_per_board != 0:
        cap.release()
        raise RuntimeError(f"Wide frame width {frame_width} is not divisible by {slots_per_board}")

    slot_width = frame_width // slots_per_board
    writers = []
    metadata = []

    for slot_index in range(slots_per_board):
        camera_index = board_index * slots_per_board + slot_index
        output_path = os.path.join(output_dir, f"cam{camera_index}.mkv")
        writer = make_writer(
            output_path=output_path,
            frame_size=(slot_width, frame_height),
            fps=fps,
            cfg_recording=cfg_recording,
        )
        if not writer.isOpened():
            cap.release()
            for opened_writer in writers:
                opened_writer.release()
            raise RuntimeError(f"Failed to create split video writer: {output_path}")

        timestamps_file = copy_timestamps(session_dir, board_summary, output_path)
        writers.append(writer)
        metadata.append({
            "camera_index": camera_index,
            "board_id": board_summary["board_id"],
            "slot_index": slot_index,
            "source_video": board_summary["video_file"],
            "output_video": os.path.basename(output_path),
            "timestamps_file": timestamps_file,
            "frame_size": {"width": slot_width, "height": frame_height},
            "fps": fps,
            "output_format": cfg_recording["output_format"],
            "jpeg_quality": cfg_recording.get("jpeg_quality"),
            "written_frames": 0,
        })

    frame_index = 0
    try:
        while ret:
            frame = ensure_gray_frame(frame, video_path)
            for slot_index, writer in enumerate(writers):
                start = slot_index * slot_width
                end = start + slot_width
                single_frame = frame[:, start:end].copy()
                writer.write(single_frame)
                metadata[slot_index]["written_frames"] += 1

            frame_index += 1
            progress_bar(board_summary["board_id"], frame_index, total_frames)
            ret, frame = cap.read()
    finally:
        cap.release()
        for writer in writers:
            writer.release()
        print()

    for item in metadata:
        info_path = os.path.join(output_dir, item["output_video"].rsplit(".", 1)[0] + "_info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(item, f, indent=2)

    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Split recorded wide-board videos into per-camera streams.")
    parser.add_argument("session_dir", help="Recorder session directory containing session_info.json")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for split camera videos. Defaults to <session_dir>/split_streams",
    )
    parser.add_argument("--slots-per-board", type=int, default=4, help="Camera slots inside each wide board video")
    return parser.parse_args()


def main():
    args = parse_args()
    session_dir = os.path.abspath(args.session_dir)
    output_dir = args.output_dir or os.path.join(session_dir, "split_streams")
    os.makedirs(output_dir, exist_ok=True)

    session_info = load_session_info(session_dir)
    boards = session_info["boards"]

    cfg_recording = {
        "output_format": "h264_mkv",
        "gst_queue_max_buffers": 8,
        "h264": {
            "encoder_impl": "x264enc",
            "bitrate": 12000,
            "gop": 30,
            "speed_preset": "ultrafast",
            "tune": "zerolatency",
            "encoder_threads": 1,
        },
        "mjpeg": {
            "jpeg_quality": 85,
        },
    }
    cfg_recording.update(session_info.get("recording", {}))
    cfg_recording = normalize_recording_config(cfg_recording)

    split_metadata = []
    for board_index, board_summary in enumerate(boards):
        split_metadata.extend(
            split_board(
                session_dir=session_dir,
                output_dir=output_dir,
                board_summary=board_summary,
                board_index=board_index,
                cfg_recording=cfg_recording,
                slots_per_board=max(1, args.slots_per_board),
            )
        )

    manifest = {
        "session_id": session_info.get("session_id"),
        "source_session_dir": session_dir,
        "output_dir": output_dir,
        "slots_per_board": args.slots_per_board,
        "streams": split_metadata,
    }
    manifest_path = os.path.join(output_dir, "split_info.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Split manifest: {manifest_path}")


if __name__ == "__main__":
    main()
