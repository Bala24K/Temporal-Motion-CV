# Fitness CV — Production Real-Time Application & iPhone Setup Guide

This guide describes how to run the **Fitness CV Real-Time Application** using an **iPhone (or external USB camera)** as the video input device while executing **100% of MediaPipe pose estimation, ML classification, rep counting, form analysis, and UI rendering locally on your Windows laptop**.

---

## 🏗️ System Architecture

```text
┌─────────────────────────┐
│     iPhone / Camera     │ (Video Input ONLY)
└────────────┬────────────┘
             │ USB / Virtual Camera Bridge (e.g. Camo, DroidCam)
             ▼
┌─────────────────────────┐
│  Windows Camera Device  │ (Enumerated as Device ID: 0, 1, 2...)
└────────────┬────────────┘
             │ OpenCV VideoCapture
             ▼
┌─────────────────────────┐
│      CameraSource       │ (Hardware Abstraction Layer)
└────────────┬────────────┘
             │ BGR Frame
             ▼
┌─────────────────────────┐
│     FitnessCVEngine     │
│ ┌─────────────────────┐ │
│ │   MediaPipe Pose    │ │
│ ├─────────────────────┤ │
│ │ Kinematic Analysis  │ │
│ ├─────────────────────┤ │
│ │  Exercise Classifier│ │
│ ├─────────────────────┤ │
│ │   Rep / Form Engine │ │
│ └─────────────────────┘ │
└────────────┬────────────┘
             │ Engine Result
             ▼
┌─────────────────────────┐
│      Dashboard UI       │ (Glassmorphism Overlay & Control State Machine)
└─────────────────────────┘
```

> **Privacy & Architectural Guarantee:**
> - **100% Local Processing:** No video frames or features are transmitted over the internet or cloud.
> - **Zero iPhone Load:** The iPhone acts strictly as an uncompressed video camera lens; no heavy ML inference or UI rendering runs on the phone.

---

## 📱 iPhone + Camo Setup

1. **Install Camo Studio on Windows:**
   - Download and install **Camo Studio for Windows** from [reincubate.com/camo](https://reincubate.com/camo/).
2. **Install Camo App on iPhone:**
   - Install **Camo** from the iOS App Store.
3. **Connect via USB:**
   - Connect your iPhone to your laptop using a USB Lightning or USB-C cable.
   - Open Camo on iOS and Camo Studio on Windows.
4. **Verify Live Feed:**
   - Confirm the HD video feed appears in Camo Studio on your laptop.

---

## 📷 Camera Selection & Calibration

### Step 1: Enumerate Camera Device IDs
Run the built-in device scanner to list connected camera indices:

```bash
python app.py --list-cameras
```

**Output:**
```text
Available Camera Devices:
  ID 0: Camera Device 0 (640x480) - Laptop Built-in Webcam
  ID 1: Camera Device 1 (1280x720) - Camo Camera (iPhone)
```

### Step 2: Diagnostic Preview Test
Test capture performance, FPS, and latency overlay:

```bash
python app.py --camera-test --camera-id 1
```

---

## 🚀 Launching the Fitness CV Application

### 1. Live Camera Mode (iPhone / Webcam)

Launch the primary application:

```bash
python app.py --camera-id 1
```

Or run via `webcam_recognizer.py`:

```bash
python webcam_recognizer.py --camera-id 1
```

### 2. Demo / Prerecorded Video Mode

Run the complete pipeline on any `.mp4` workout video file:

```bash
python app.py --demo-video path/to/workout_clip.mp4
```

---

## 🎮 Interactive Keyboard Controls

| Key | Action |
| :--- | :--- |
| **`[SPACE]`** | Start / Pause workout session (or transition from Setup to Running) |
| **`[P]`** | Pause / Unpause session timer and rep counter |
| **`[R]`** | Reset rep count and session metrics |
| **`[M]`** | Cycle exercise mode (`AUTO DETECT` ➔ `BICEP CURL` ➔ `SQUAT` ➔ `SHOULDER PRESS` ➔ `LATERAL RAISE`) |
| **`[D]`** | Toggle diagnostic debug overlay (landmark IDs & raw angles) |
| **`[F]`** | Toggle horizontal mirror flip |
| **`[S]`** | Save session summary to local JSON file (in Summary View) |
| **`[E]` / `[Q]`** | End session & open Session Summary view / Exit |

---

## 📐 Recommended Camera Placement

1. **Distance:** Stand **2 to 3 meters (6–10 feet)** back from the iPhone camera.
2. **Height:** Mount or rest your iPhone at approximately chest or waist height.
3. **Framing:** Ensure your **head, torso, hips, knees, and feet** are fully visible inside the camera setup guide box.
4. **Lighting:** Ensure good front or side lighting (avoid strong backlighting).

---

## ⚙️ CLI Argument Reference

| Option | Description | Default |
| :--- | :--- | :--- |
| `--camera-id <INT>` | Select video capture device ID (e.g. `0` for webcam, `1` for iPhone) | `0` |
| `--list-cameras` | Enumerate available Windows video capture devices | `False` |
| `--camera-test` | Run diagnostic camera feed window | `False` |
| `--demo-video <PATH>`| Path to `.mp4` video file for deterministic demo mode | `None` |
| `--exercise <NAME>` | Exercise mode (`AUTO`, `BICEP CURL`, `SQUAT`, `SHOULDER PRESS`, `LATERAL RAISE`) | `AUTO` |
| `--width <INT>` | Requested frame width | `1280` |
| `--height <INT>` | Requested frame height | `720` |
| `--fps <INT>` | Requested capture frame rate | `30` |
| `--debug-overlay` | Display diagnostic pose landmarks and joint angles | `False` |
| `--record-session` | Write session output video to local `recordings/` directory | `False` |

---

## 📊 Session Summary & JSON Export

When a session is ended (`[E]`), the Session Summary screen displays:
- Exercise Mode & Detected Exercise
- Total Reps Completed
- Active Session Duration (`MM:SS`)
- Average Confidence %
- Average System FPS & Latency
- Form Quality Summary (Good vs Warnings)

Press **`[S]`** to export the summary to a local JSON file in `sessions/session_YYYYMMDD_HHMMSS.json`.
