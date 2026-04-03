# Bag Extraction Script

`extract_bags.py` reads ROS `.bag` files from a single dive, extracts sensor data and GPS readings into Supabase, and converts the camera footage into video clips plus one full-dive video stored in Supabase Storage.

---

## What it does

| Output | Where |
|--------|-------|
| Per-bag video clips (`clip_000.mp4` … `clip_052.mp4`) | Supabase Storage → `dives/{dive_id}/clips/` |
| Full-dive video (`full_dive.mp4`) | Supabase Storage → `dives/{dive_id}/` |
| Sensor readings (IMU, pressure, temperature, sonar, voltage, leak, magnetometer) | `sensor_readings` table |
| GPS fixes | `gps_readings` table |
| Dive session record | `dives` table |

---

## Prerequisites

### 1 — Create the Supabase tables

Open your Supabase dashboard → **SQL Editor** and run:

```sql
CREATE TABLE dives (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    recorded_at TIMESTAMPTZ,
    bag_dir     TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE sensor_readings (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dive_id          UUID REFERENCES dives(id) ON DELETE CASCADE,
    recorded_at      TIMESTAMPTZ NOT NULL,
    ros_time         DOUBLE PRECISION,
    pressure_pa      DOUBLE PRECISION,
    water_temp_c     DOUBLE PRECISION,
    sonar_dist_m     DOUBLE PRECISION,
    voltage          DOUBLE PRECISION,
    leak_detected    BOOLEAN,
    imu_orient_x     DOUBLE PRECISION,
    imu_orient_y     DOUBLE PRECISION,
    imu_orient_z     DOUBLE PRECISION,
    imu_orient_w     DOUBLE PRECISION,
    imu_ang_vel_x    DOUBLE PRECISION,
    imu_ang_vel_y    DOUBLE PRECISION,
    imu_ang_vel_z    DOUBLE PRECISION,
    imu_lin_acc_x    DOUBLE PRECISION,
    imu_lin_acc_y    DOUBLE PRECISION,
    imu_lin_acc_z    DOUBLE PRECISION,
    mag_x            DOUBLE PRECISION,
    mag_y            DOUBLE PRECISION,
    mag_z            DOUBLE PRECISION
);

CREATE TABLE gps_readings (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dive_id     UUID REFERENCES dives(id) ON DELETE CASCADE,
    recorded_at TIMESTAMPTZ NOT NULL,
    ros_time    DOUBLE PRECISION,
    latitude    DOUBLE PRECISION,
    longitude   DOUBLE PRECISION,
    altitude    DOUBLE PRECISION,
    fix_status  INTEGER
);

CREATE TABLE dive_videos (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dive_id      UUID REFERENCES dives(id) ON DELETE CASCADE,
    video_type   TEXT NOT NULL,
    bag_filename TEXT,
    clip_index   INTEGER,
    storage_path TEXT NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
```

### 2 — Install Python packages

With your virtual environment active:

```powershell
pip install supabase==2.15.2 opencv-python==4.11.0.86
```

### 3 — Install ffmpeg

ffmpeg is required to merge the clips into the full-dive video.

- **Windows:** Download from https://ffmpeg.org/download.html and add the `bin/` folder to your PATH
- **Linux/macOS:** `sudo apt install ffmpeg` or `brew install ffmpeg`

Verify it works:
```bash
ffmpeg -version
```

### 4 — ROS environment

The script uses `rosbag` which is part of ROS. Run the script inside a ROS environment where `import rosbag` works.

---

## Running the script

### Step 1 — Set your bag folder path

Open `scripts/extract_bags.py` and edit line 45:

```python
BAG_DIR = r"C:\path\to\your\bag\folder"   # ← change this
```

Replace with the actual path to the folder containing your `.bag` files, e.g.:

```python
BAG_DIR = r"C:\Users\mingx\bags\EddyReef2_2025-11-09"
```

### Step 2 — Run it

From the project root, with your virtual environment active:

```powershell
python scripts/extract_bags.py
```

### Step 3 — Watch the output

The script prints progress for each bag:

```
Found 53 bag files in C:\...

✓ Dive created  id=abc123...  name=EddyReef2_2025-11-09

[00/52] all-EddyReef2_2025-11-09-04-24-46_0.bag
  video  : 312 frames → clip_000.mp4
  sensors: 1540 rows inserted
  gps    : 30 rows inserted
  upload : dives/abc123.../clips/clip_000.mp4

[01/52] all-EddyReef2_2025-11-09-04-25-16_1.bag
  ...

Merging clips into full_dive.mp4 ...
✓ Uploaded full_dive.mp4 → dives/abc123.../full_dive.mp4

✓ Done.  Dive ID: abc123...
```

---

## Verifying the results

After the script finishes:

1. **Supabase Table Editor** → `dives` — should have 1 row
2. **Supabase Table Editor** → `sensor_readings` — should have thousands of rows
3. **Supabase Table Editor** → `gps_readings` — should have hundreds of rows
4. **Supabase Table Editor** → `dive_videos` — should have 53 clip rows + 1 full row
5. **Supabase Storage** → `dive-videos` bucket → `dives/{dive_id}/` — should contain `clips/` folder and `full_dive.mp4`
