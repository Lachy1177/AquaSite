#!/usr/bin/env python3
"""
extract_bags.py — Extract ROS bag data and upload to Supabase.

See scripts/README.md for setup and usage instructions.
"""

import os
import re
import sys
import glob
import subprocess
import tempfile
from datetime import datetime, timezone

import cv2
import numpy as np

# ── Locate project root and load .env ─────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def load_env(env_path):
    """Parse a .env file and populate os.environ with its values."""
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


load_env(os.path.join(PROJECT_ROOT, ".env"))

# ── Configuration ──────────────────────────────────────────────────────────────

BAG_DIR = r"C:\path\to\your\bag\folder"   # ← EDIT THIS before running

DIVE_NAME   = "EddyReef2_2025-11-09"      # Human-readable label for this dive
CAMERA_FPS  = 10    # Frames per second to use when writing video clips
BATCH_SIZE  = 500   # Max rows per Supabase insert call
TEMP_DIR    = tempfile.mkdtemp(prefix="aquasight_")

SUPABASE_URL    = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY    = os.environ.get("SUPABASE_KEY", "")
STORAGE_BUCKET  = os.environ.get("SUPABASE_STORAGE_BUCKET", "dive-videos")

# ── ROS topic names ────────────────────────────────────────────────────────────

TOPIC_IMU      = "/glider_1/sensor/imu"
TOPIC_PRESSURE = "/glider_1/sensor/pressure"
TOPIC_TEMP     = "/glider_1/sensor/pressure/temperature/water"
TOPIC_SONAR    = "/glider_1/sensor/sonar_down/distance"
TOPIC_VOLTAGE  = "/glider_1/sensor/voltage/beaglebone_input"
TOPIC_LEAK     = "/glider_1/sensor/leak"
TOPIC_MAG      = "/glider_1/sensor/mag"
TOPIC_CAMERA   = "/glider_1/camera_front/image_raw/compressed"
TOPIC_GPS      = "/surface/ublox_gps/fix"

SENSOR_TOPICS = [
    TOPIC_IMU, TOPIC_PRESSURE, TOPIC_TEMP, TOPIC_SONAR,
    TOPIC_VOLTAGE, TOPIC_LEAK, TOPIC_MAG,
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def ros_time_to_iso(t_ns):
    """Convert a rosbags nanosecond timestamp to a UTC ISO-8601 string."""
    return datetime.fromtimestamp(t_ns / 1e9, tz=timezone.utc).isoformat()


def bag_sort_key(path):
    """Sort all-*.bag by numeric suffix (_0 … _52)."""
    m = re.search(r"_(\d+)\.bag$", path)
    return int(m.group(1)) if m else -1


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def batch_insert(supabase, table, rows):
    for batch in chunked(rows, BATCH_SIZE):
        supabase.table(table).insert(batch).execute()


def upload_to_storage(supabase, local_path, storage_path):
    with open(local_path, "rb") as f:
        data = f.read()
    supabase.storage.from_(STORAGE_BUCKET).upload(
        storage_path,
        data,
        file_options={"content-type": "video/mp4"},
    )

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Validate config
    if BAG_DIR == r"C:\path\to\your\bag\folder":
        print("ERROR: Set BAG_DIR in this script before running.")
        sys.exit(1)
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: SUPABASE_URL and SUPABASE_KEY must be set in your .env file.")
        sys.exit(1)

    try:
        from rosbags.rosbag1 import Reader
        from rosbags.typesys import Stores, get_typestore
    except ImportError:
        print("ERROR: rosbags not found. Run: pip install rosbags")
        sys.exit(1)

    from supabase import create_client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Find and sort bags
    bags = sorted(glob.glob(os.path.join(BAG_DIR, "all-*.bag")), key=bag_sort_key)
    if not bags:
        print(f"ERROR: No all-*.bag files found in {BAG_DIR}")
        sys.exit(1)
    print(f"Found {len(bags)} bag files in {BAG_DIR}\n")

    # Build typestore once; register custom message types from the first bag
    typestore = get_typestore(Stores.ROS1_NOETIC)
    _registered_types = set()

    def register_bag_types(reader):
        try:
            from rosbags.typesys import get_types_from_msg
            add_types = {}
            for conn in reader.connections:
                if conn.msgtype not in _registered_types:
                    add_types.update(get_types_from_msg(conn.msgdef, conn.msgtype))
                    _registered_types.add(conn.msgtype)
            if add_types:
                typestore.register(add_types)
        except Exception:
            pass  # standard types will still work

    # Detect dive start time from first bag
    with Reader(bags[0]) as b:
        register_bag_types(b)
        dive_start = datetime.fromtimestamp(b.start_time / 1e9, tz=timezone.utc)

    # Create the dive record
    result = supabase.table("dives").insert({
        "name": DIVE_NAME,
        "recorded_at": dive_start.isoformat(),
        "bag_dir": BAG_DIR,
    }).execute()
    dive_id = result.data[0]["id"]
    print(f"✓ Dive created  id={dive_id}  name={DIVE_NAME}\n")

    clip_paths = []

    for i, bag_path in enumerate(bags):
        bag_name = os.path.basename(bag_path)
        print(f"[{i:02d}/{len(bags)-1}] {bag_name}")

        clip_path = os.path.join(TEMP_DIR, f"clip_{i:03d}.mp4")
        sensor_rows = []
        gps_rows = []

        # Last-known values for slow-publishing sensors
        state = {
            "pressure_pa":   None,
            "water_temp_c":  None,
            "sonar_dist_m":  None,
            "voltage":       None,
            "leak_detected": None,
            "mag_x": None, "mag_y": None, "mag_z": None,
        }

        video_writer = None
        frame_count = 0

        read_topics = set([TOPIC_CAMERA] + SENSOR_TOPICS + [TOPIC_GPS])

        with Reader(bag_path) as bag:
            register_bag_types(bag)
            connections = [c for c in bag.connections if c.topic in read_topics]
            for connection, timestamp, rawdata in bag.messages(connections=connections):
                topic = connection.topic
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                dt_str = ros_time_to_iso(timestamp)
                ros_time = timestamp / 1e9

                # ── Camera frames ──────────────────────────────────────────
                if topic == TOPIC_CAMERA:
                    raw = msg.data
                    arr = np.frombuffer(bytes(raw) if not isinstance(raw, (bytes, bytearray)) else raw, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    if video_writer is None:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(
                            clip_path, fourcc, CAMERA_FPS, (w, h)
                        )
                    video_writer.write(frame)
                    frame_count += 1

                # ── Slow sensors (carry-forward state) ─────────────────────
                elif topic == TOPIC_PRESSURE:
                    state["pressure_pa"] = msg.fluid_pressure

                elif topic == TOPIC_TEMP:
                    state["water_temp_c"] = msg.temperature

                elif topic == TOPIC_SONAR:
                    # glider/msg/SonarDistance — try both field name variants
                    state["sonar_dist_m"] = (
                        getattr(msg, "distance", None)
                        or getattr(msg, "range", None)
                    )

                elif topic == TOPIC_VOLTAGE:
                    state["voltage"] = msg.data

                elif topic == TOPIC_LEAK:
                    # glider/msg/Leak — try both field name variants
                    raw = getattr(msg, "detected", None)
                    if raw is None:
                        raw = getattr(msg, "leak", None)
                    state["leak_detected"] = bool(raw) if raw is not None else None

                elif topic == TOPIC_MAG:
                    state["mag_x"] = msg.magnetic_field.x
                    state["mag_y"] = msg.magnetic_field.y
                    state["mag_z"] = msg.magnetic_field.z

                # ── IMU — one row per message, snapshot current state ──────
                elif topic == TOPIC_IMU:
                    sensor_rows.append({
                        "dive_id":      dive_id,
                        "recorded_at":  dt_str,
                        "ros_time":     ros_time,
                        "imu_orient_x": msg.orientation.x,
                        "imu_orient_y": msg.orientation.y,
                        "imu_orient_z": msg.orientation.z,
                        "imu_orient_w": msg.orientation.w,
                        "imu_ang_vel_x": msg.angular_velocity.x,
                        "imu_ang_vel_y": msg.angular_velocity.y,
                        "imu_ang_vel_z": msg.angular_velocity.z,
                        "imu_lin_acc_x": msg.linear_acceleration.x,
                        "imu_lin_acc_y": msg.linear_acceleration.y,
                        "imu_lin_acc_z": msg.linear_acceleration.z,
                        **state,
                    })

                # ── GPS ────────────────────────────────────────────────────
                elif topic == TOPIC_GPS:
                    gps_rows.append({
                        "dive_id":    dive_id,
                        "recorded_at": dt_str,
                        "ros_time":   ros_time,
                        "latitude":   msg.latitude,
                        "longitude":  msg.longitude,
                        "altitude":   msg.altitude,
                        "fix_status": int(msg.status.status),
                    })

        # Finalise video clip
        if video_writer:
            video_writer.release()
            clip_paths.append(clip_path)
            print(f"  video  : {frame_count} frames → {os.path.basename(clip_path)}")
        else:
            clip_paths.append(None)
            print(f"  video  : no camera frames in this bag")

        # Insert to Supabase
        if sensor_rows:
            batch_insert(supabase, "sensor_readings", sensor_rows)
            print(f"  sensors: {len(sensor_rows)} rows inserted")
        if gps_rows:
            batch_insert(supabase, "gps_readings", gps_rows)
            print(f"  gps    : {len(gps_rows)} rows inserted")

        # Upload clip
        valid_clip = (
            video_writer
            and os.path.exists(clip_path)
            and os.path.getsize(clip_path) > 0
        )
        if valid_clip:
            storage_path = f"dives/{dive_id}/clips/clip_{i:03d}.mp4"
            upload_to_storage(supabase, clip_path, storage_path)
            supabase.table("dive_videos").insert({
                "dive_id":     dive_id,
                "video_type":  "clip",
                "bag_filename": bag_name,
                "clip_index":  i,
                "storage_path": storage_path,
            }).execute()
            print(f"  upload : {storage_path}")

        print()

    # ── Merge all clips into one full-dive video ───────────────────────────────
    valid_clips = [
        p for p in clip_paths
        if p and os.path.exists(p) and os.path.getsize(p) > 0
    ]

    if valid_clips:
        filelist_path = os.path.join(TEMP_DIR, "filelist.txt")
        with open(filelist_path, "w") as f:
            for cp in valid_clips:
                # ffmpeg requires forward slashes even on Windows
                f.write(f"file '{cp.replace(chr(92), '/')}'\n")

        full_video_path = os.path.join(TEMP_DIR, "full_dive.mp4")
        print("Merging clips into full_dive.mp4 ...")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", filelist_path,
                "-c", "copy",
                full_video_path,
            ],
            check=True,
        )

        storage_path = f"dives/{dive_id}/full_dive.mp4"
        upload_to_storage(supabase, full_video_path, storage_path)
        supabase.table("dive_videos").insert({
            "dive_id":     dive_id,
            "video_type":  "full",
            "bag_filename": None,
            "clip_index":  None,
            "storage_path": storage_path,
        }).execute()
        print(f"✓ Uploaded full_dive.mp4 → {storage_path}")
    else:
        print("No valid clips found — skipping full video merge.")

    print(f"\n✓ Done.  Dive ID: {dive_id}")


if __name__ == "__main__":
    main()
