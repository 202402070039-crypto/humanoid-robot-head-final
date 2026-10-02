"""
============================================================
ROBOT HEAD - FINAL VOICE + CAMERA INTEGRATION
============================================================

VOICE SYSTEM:
- Multilingual STT: English / Hindi / Marathi
- Wake word
- Conversation
- AI response
- TTS
- Jaw audio synchronization
- Conversation history
- Face enrollment

COMPUTER VISION:
- YOLO people detection
- People counting
- Closest-person target selection
- Stable target ID
- YOLO object detection
- MediaPipe face landmarks
- Head horizontal direction
- Head vertical direction
- Head tilt
- Gaze
- Person position information for AI

IMPORTANT:
- Eye/neck servo control is driven by camera face position.
- Jaw servo remains driven by audio amplitude.
- MQTT is used for all four servo commands.
- Camera preview is kept for testing.
- Camera processing continues in background while voice is active.
- Robot-relative LEFT/RIGHT mapping is corrected for the mirrored preview.
- CV processing is adaptively reduced during voice processing to reduce CPU contention.
- The latest valid CV state is retained while CV runs at the reduced rate.
- Timing diagnostics are printed for STT, AI, TTS, and jaw/audio playback.
============================================================
"""

import os
import re
import json
import time
import math
import threading
from collections import OrderedDict

import cv2
import numpy as np

from insightface.app import FaceAnalysis
from ultralytics import YOLO

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import speech_recognition
import robot_conversation
import text_to_speech
import jaw_audio_sync
import hardware_control
import robot_self_knowledge
# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

KNOWN_FACES_DIR = os.path.join(
    BASE_DIR,
    "known_faces"
)

CONVERSATION_DIR = os.path.join(
    BASE_DIR,
    "conversation_history1"
)

YOLO_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "yolo11n.pt"
)

FACE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "face_landmarker.task"
)

os.makedirs(
    KNOWN_FACES_DIR,
    exist_ok=True
)

os.makedirs(
    CONVERSATION_DIR,
    exist_ok=True
)


# ============================================================
# CAMERA SETTINGS
# ============================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


# ============================================================
# COMPUTER VISION PERFORMANCE
# ============================================================

# Normal CV frequency
YOLO_INTERVAL = 5
FACE_INTERVAL = 2
FACE_RECOGNITION_INTERVAL = 12

# Reduced CV frequency while voice is actively processing.
# The most recent valid CV state is retained, so the AI
# can still answer camera questions during voice processing.
YOLO_INTERVAL_VOICE = 15
FACE_INTERVAL_VOICE = 6
FACE_RECOGNITION_INTERVAL_VOICE = 30

YOLO_CONFIDENCE = 0.40

FACE_DETECTION_CONFIDENCE = 0.50
FACE_PRESENCE_CONFIDENCE = 0.50
FACE_TRACKING_CONFIDENCE = 0.50


# ============================================================
# FACE RECOGNITION
# ============================================================

RECOGNITION_THRESHOLD = 0.45


# ============================================================
# HEAD DIRECTION
# ============================================================

HORIZONTAL_DEADZONE = 0.12
VERTICAL_DEADZONE = 0.12

TILT_DEADZONE = 8.0


# ============================================================
# SERVO CONTROL - CAMERA TRACKING
# ============================================================
#
# Camera tracking uses:
#   MediaPipe face center
#       -> normalized position error
#       -> EMA face smoothing
#       -> constant-speed servo movement
#
# Jaw is NOT controlled here. jaw_audio_sync remains responsible
# for jaw movement during speech.
#
# Eye calibration:
#   Horizontal: 45 RIGHT / 90 CENTER / 135 LEFT
#   Vertical:   90 UP / 135 CENTER / 180 DOWN
#
# Neck:
#   90 CENTER / 180 currently established safe end.
#   The direction is isolated in NECK_IMAGE_RIGHT_SIGN because
#   the physical neck direction must be verified on the assembled
#   mechanism.
# ============================================================

SERVO_CONTROL_INTERVAL = 0.05

EYE_H_MIN = 45.0
EYE_H_CENTER = 90.0
EYE_H_MAX = 135.0

EYE_V_MIN = 90.0
EYE_V_CENTER = 135.0
EYE_V_MAX = 180.0

NECK_MIN = 0.0
NECK_CENTER = 90.0
NECK_MAX = 180.0

# Based on the previous neck test, increasing angle follows
# image-right. Change to -1.0 only after physical verification.
NECK_IMAGE_RIGHT_SIGN = 1.0

SERVO_DEADZONE = 0.07

# ============================================================
# STABLE FACE TRACKING
# ============================================================
# The camera is mounted in the moving head. A direct
# face-X -> neck-angle mapping creates a feedback loop:
# neck moves -> camera view changes -> detected face moves ->
# neck is commanded again. The controller below therefore uses
# hierarchical tracking and incremental neck motion.
#
# FIX NOTES (root causes addressed by the values/logic below):
#   1. Two-point eye oscillation was caused by recomputing the eye
#      target every tick with no confirmation, and by force-snapping
#      the horizontal eye target every single tick while the neck was
#      engaged instead of only once, at the hand-off transition.
#   2. Unwanted vertical motion was caused by not accounting for the
#      fact that neck (and eye) rotation itself shifts the face's
#      apparent Y position in frame (parallax from the moving camera).
#   3. Overshoot/"whip" past the face on one side only was caused by
#      the original neck hand-off condition being ONE-SIDED. The
#      symmetric, head-relative check below (see note 4) also fixes
#      this, since it is identical for both directions by construction.
#   4. The neck displaying "hand-off ACTIVE" repeatedly while barely
#      moving, and the eyes visibly reaching center BEFORE the neck
#      caught up (looking like two separate sequential actions instead
#      of one simultaneous motion), were both caused by the same root
#      issue: the camera is mounted INSIDE THE EYE, not just the neck.
#      When the eyes recenter, the camera physically swings too -- so
#      the image-based error signal collapses almost immediately from
#      the eye's own motion, not from the neck actually closing the
#      gap. That made the controller think the target was reached and
#      release the neck long before it had moved far, then re-engage
#      moments later as the true (uncompensated) error reappeared --
#      exactly the ACTIVE/RELEASED flicker seen in testing.
#
#      FIX: the neck engage/release decision now uses a HEAD-RELATIVE
#      angle -- current eye deflection from center PLUS the remaining
#      image-space offset -- instead of the raw image position alone.
#      This total stays accurate no matter where the eyes currently
#      are pointed, because as the eyes move toward center the image
#      offset grows by the same amount the eye deflection shrinks.
#      The eyes are also given a slower, dedicated recenter speed
#      (EYE_RECENTER_SPEED) during hand-off instead of their normal
#      fast tracking speed, so the eye-return and the neck motion
#      finish at roughly the same time and read as one continuous
#      motion instead of two sequential ones.

FACE_EMA_ALPHA_X = 0.20
FACE_EMA_ALPHA_Y = 0.12

# Face-position hysteresis, kept for compatibility with any other
# module that references these. The active pixel-noise filtering is
# done via EYE_TARGET_HYSTERESIS_DEGREES / EYE_TARGET_V_HYSTERESIS_DEGREES
# below, which operate in servo-degree space after mapping.
EYE_HORIZONTAL_HYSTERESIS_PIXELS = 14.0
EYE_VERTICAL_HYSTERESIS_PIXELS = 26.0
EYE_CONFIRM_FRAMES = 4

# While the neck is rotating -- and for a short cool-down window after
# it stops -- camera-induced apparent motion must not move the servos.
# This is applied to BOTH axes now, not just vertical, because the
# neck's own rotation also disturbs the horizontal reading briefly.
NECK_VERTICAL_HOLD_SECONDS = 0.40
NECK_HORIZONTAL_HOLD_SECONDS = 0.25

# Eyes track first. The neck starts only when the face's HEAD-RELATIVE
# angle (eye deflection + remaining image offset -- see FIX NOTE 4
# above) exceeds NECK_START_DEG, and releases once that same
# compensated angle falls back inside NECK_STOP_DEG. Both are true
# degrees, symmetric for either direction, and immune to the eyes'
# own recentering motion.
NECK_START_DEG = 18.0   # engage once compensated head-relative angle exceeds this
NECK_STOP_DEG = 5.0     # release once it falls back inside this (~5 degree tolerance)
NECK_FULL_SPEED_DEG = 45.0  # compensated angle at which the neck reaches max speed
NECK_CONFIRM_FRAMES = 6
NECK_RELEASE_FRAMES = 8

# Extra eye-target hysteresis. The eyes do not chase every tiny change
# in the detected face position -- a candidate target must persist in
# the same direction for EYE_CONFIRM_FRAMES before it is accepted.
EYE_TARGET_HYSTERESIS_DEGREES = 4.0
EYE_TARGET_V_HYSTERESIS_DEGREES = 5.0

# Neck speed: proportional to the remaining compensated error, updated
# EVERY servo loop tick (not throttled to a slower command interval).
# A fixed command interval (e.g. 10 Hz) makes the neck visibly "step"
# from one commanded position to the next between updates instead of
# gliding continuously -- this was the "hanging on one increment point
# then another" symptom. Updating every tick at the full loop rate
# removes that stepping.
NECK_TRACK_SPEED_MIN = 3.0
NECK_TRACK_SPEED_MAX = 8.0

# DERIVATIVE BRAKING (ported from the version that tracked smoothly):
# instead of relying only on a fixed speed ramp, measure how fast the
# compensated error is ALREADY shrinking each tick. If it's closing in
# quickly (the neck + face motion is naturally converging), cut speed
# further so the neck doesn't sail past the stop point and have to
# correct back -- this is what removes the "overmoves, then takes 2-3
# rounds to settle" pattern for larger movements.
NECK_ERROR_RATE_FAST_DEG = 20.0   # deg/sec closing rate -> heavy brake
NECK_ERROR_RATE_MED_DEG = 8.0     # deg/sec closing rate -> light brake
NECK_BRAKE_FAST_FACTOR = 0.55
NECK_BRAKE_MED_FACTOR = 0.78

# Normal small-movement eye tracking speed (neck NOT engaged).
EYE_MAX_SPEED = 65.0

# Dedicated, slower speed used ONLY while the neck is actively engaged
# and the eyes are recentering. Deliberately paced close to the neck's
# typical closing speed (NECK_TRACK_SPEED_MIN..MAX) so the eye-return
# and the neck rotation finish together instead of the eyes visibly
# arriving first.
EYE_RECENTER_SPEED = 6.0

NECK_RETURN_SPEED = 10.0
NECK_MAX_SPEED = 18.0

NECK_ACTIVATION = 0.20

# FACE_STALE_TIMEOUT: data newer than this is treated as "fresh" and is
# actively used to steer the eyes/neck.
#
# FACE_LOST_RESET_TIMEOUT: only once data is OLDER than this do we treat
# the face as genuinely gone and reset (release the neck, recenter the
# eyes, clear the smoothing filters). Between the two timeouts the data
# is "stale but not lost yet" -- the controller HOLDS its current state
# and does nothing, instead of resetting.
#
# This gap matters a lot in practice: heavy CPU load elsewhere in the
# process (STT, LLM calls, TTS generation, jaw sync, face recognition,
# YOLO) can delay the CV thread's landmark updates well past a tight
# stale timeout even though the face never actually left the frame. A
# single timeout that resets on every stale tick causes the neck
# hand-off to restart from zero constantly -- it never accumulates
# enough continuous engagement to actually move, and the eyes keep
# re-snapping to a fresh (unsmoothed) position every time, which looks
# like drifting/wandering even while the person is standing still.
FACE_STALE_TIMEOUT = 0.45
FACE_LOST_RESET_TIMEOUT = 1.50


# ============================================================
# TARGET TRACKING
# ============================================================

TARGET_MATCH_DISTANCE = 120

TARGET_MEMORY_FRAMES = 30


# ============================================================
# VOICE SETTINGS
# ============================================================

MAX_TTS_CHARACTERS = 400

MAX_HISTORY_MESSAGES = 4


# ============================================================
# WAKE WORDS
# ============================================================

WAKE_WORDS = [
    "master",
    "hey master",
    "hi master",
    "hello master"
]


# ============================================================
# EXIT WORDS
# ============================================================

EXIT_WORDS = [
    "exit",
    "stop",
    "bye",
    "goodbye",
    "quit"
]


# ============================================================
# SHARED CAMERA DATA
# ============================================================

latest_frame = None

camera_lock = threading.Lock()


# ============================================================
# CV STATE
# ============================================================

cv_people = []

cv_targets = []

cv_objects = []

cv_face_data = None
cv_face_timestamp = 0.0

cv_selected_target = None

cv_fps = 0.0

cv_state_lock = threading.Lock()


# ============================================================
# CURRENT RECOGNIZED PERSON
# ============================================================

current_person = "UNKNOWN"

current_similarity = 0.0

current_person_lock = threading.Lock()


# ============================================================
# KNOWN FACE DATABASE
# ============================================================

known_faces = {}

known_faces_lock = threading.Lock()


# ============================================================
# THREAD CONTROL
# ============================================================

running = True

voice_processing = False

enrollment_active = False


# ============================================================
# SERVO STATE
# ============================================================

servo_control_running = True

servo_lock = threading.Lock()

current_eye_horizontal = EYE_H_CENTER
current_eye_vertical = EYE_V_CENTER
current_neck = NECK_CENTER


# ============================================================
# ============================================================
# CAMERA STREAM
# ============================================================
# ============================================================

class CameraStream:

    def __init__(
        self,
        index=0
    ):

        self.camera = cv2.VideoCapture(
            index,
            cv2.CAP_DSHOW
        )

        if not self.camera.isOpened():

            raise RuntimeError(
                "Could not open camera."
            )

        self.camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            FRAME_WIDTH
        )

        self.camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            FRAME_HEIGHT
        )

        self.camera.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        self.running = True

        self.lock = threading.Lock()

        self.latest_frame = None

        self.thread = threading.Thread(
            target=self.update,
            daemon=True
        )

        self.thread.start()

        print(
            "Camera started successfully."
        )

    def update(self):

        global latest_frame

        while self.running:

            ret, frame = (
                self.camera.read()
            )

            if not ret:

                time.sleep(
                    0.01
                )

                continue

            with self.lock:

                self.latest_frame = frame

            with camera_lock:

                latest_frame = frame

        print(
            "Camera capture stopped."
        )

    def read(self):

        with self.lock:

            if self.latest_frame is None:

                return None

            return self.latest_frame.copy()

    def stop(self):

        self.running = False

        if self.thread.is_alive():

            self.thread.join(
                timeout=1
            )

        self.camera.release()


# ============================================================
# ============================================================
# STABLE TARGET TRACKER
# ============================================================
# ============================================================

class StableTargetTracker:

    def __init__(self):

        self.next_id = 1

        self.targets = OrderedDict()

    def center_of_box(
        self,
        box
    ):

        x1, y1, x2, y2 = box

        return (
            (x1 + x2) // 2,
            (y1 + y2) // 2
        )

    def distance(
        self,
        p1,
        p2
    ):

        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )

    def update(
        self,
        detections
    ):

        current_frame_targets = []

        used_old_ids = set()

        for detection in detections:

            box = detection["box"]

            current_center = (
                self.center_of_box(box)
            )

            best_id = None

            best_distance = (
                TARGET_MATCH_DISTANCE
            )

            for (
                target_id,
                old_target
            ) in self.targets.items():

                if target_id in used_old_ids:

                    continue

                d = self.distance(
                    current_center,
                    old_target["center"]
                )

                if d < best_distance:

                    best_distance = d

                    best_id = target_id

            if best_id is not None:

                target_id = best_id

                used_old_ids.add(
                    target_id
                )

            else:

                target_id = self.next_id

                self.next_id += 1

            self.targets[target_id] = {

                "center":
                    current_center,

                "box":
                    box,

                "confidence":
                    detection["confidence"],

                "missing":
                    0
            }

            current_frame_targets.append({

                "id":
                    target_id,

                "box":
                    box,

                "center":
                    current_center,

                "confidence":
                    detection["confidence"]
            })

        visible_ids = {
            target["id"]
            for target in
            current_frame_targets
        }

        remove_ids = []

        for target_id in list(
            self.targets.keys()
        ):

            if target_id not in visible_ids:

                self.targets[
                    target_id
                ]["missing"] += 1

                if (
                    self.targets[
                        target_id
                    ]["missing"]
                    >
                    TARGET_MEMORY_FRAMES
                ):

                    remove_ids.append(
                        target_id
                    )

        for target_id in remove_ids:

            del self.targets[
                target_id
            ]

        return current_frame_targets


# ============================================================
# ============================================================
# MEDIAPIPE FACE PROCESSOR
# ============================================================
# ============================================================

class FaceProcessor:

    def __init__(self):

        base_options = (
            python.BaseOptions(
                model_asset_path=
                FACE_MODEL_PATH
            )
        )

        options = (
            vision.FaceLandmarkerOptions(

                base_options=
                    base_options,

                running_mode=
                    vision.RunningMode.VIDEO,

                num_faces=1,

                min_face_detection_confidence=
                    FACE_DETECTION_CONFIDENCE,

                min_face_presence_confidence=
                    FACE_PRESENCE_CONFIDENCE,

                min_tracking_confidence=
                    FACE_TRACKING_CONFIDENCE,

                output_face_blendshapes=False,

                output_facial_transformation_matrixes=
                    False
            )
        )

        self.landmarker = (
            vision.FaceLandmarker
            .create_from_options(
                options
            )
        )

        self.timestamp_ms = 0

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    def calculate_direction(
        self,
        face_x,
        face_y,
        frame_width,
        frame_height
    ):

        normalized_x = (
            face_x /
            frame_width
        )

        normalized_y = (
            face_y /
            frame_height
        )

        # Camera preview is treated as mirrored.
        # Therefore image-left corresponds to the robot's RIGHT
        # and image-right corresponds to the robot's LEFT.

        if (
            normalized_x
            <
            0.5 -
            HORIZONTAL_DEADZONE
        ):

            horizontal = "RIGHT"

        elif (
            normalized_x
            >
            0.5 +
            HORIZONTAL_DEADZONE
        ):

            horizontal = "LEFT"

        else:

            horizontal = "CENTER"

        if (
            normalized_y
            <
            0.5 -
            VERTICAL_DEADZONE
        ):

            vertical = "UP"

        elif (
            normalized_y
            >
            0.5 +
            VERTICAL_DEADZONE
        ):

            vertical = "DOWN"

        else:

            vertical = "CENTER"

        return (
            horizontal,
            vertical
        )

    # --------------------------------------------------------
    # Tilt
    # --------------------------------------------------------

    def calculate_tilt(
        self,
        landmarks
    ):

        try:

            left_eye = landmarks[33]

            right_eye = landmarks[263]

            dx = (
                right_eye.x -
                left_eye.x
            )

            dy = (
                right_eye.y -
                left_eye.y
            )

            angle = math.degrees(
                math.atan2(
                    dy,
                    dx
                )
            )

            if angle > 90:

                angle -= 180

            if angle < -90:

                angle += 180

            return angle

        except Exception:

            return 0.0

    # --------------------------------------------------------
    # Gaze
    # --------------------------------------------------------

    def calculate_gaze(
        self,
        landmarks
    ):

        try:

            left_iris_points = (
                landmarks[468:473]
            )

            left_iris_x = (
                sum(
                    p.x
                    for p in
                    left_iris_points
                )
                /
                len(
                    left_iris_points
                )
            )

            right_iris_points = (
                landmarks[473:478]
            )

            right_iris_x = (
                sum(
                    p.x
                    for p in
                    right_iris_points
                )
                /
                len(
                    right_iris_points
                )
            )

            left_outer = (
                landmarks[33].x
            )

            left_inner = (
                landmarks[133].x
            )

            right_inner = (
                landmarks[362].x
            )

            right_outer = (
                landmarks[263].x
            )

            left_ratio = (
                left_iris_x -
                left_outer
            ) / (
                left_inner -
                left_outer +
                1e-6
            )

            right_ratio = (
                right_iris_x -
                right_inner
            ) / (
                right_outer -
                right_inner +
                1e-6
            )

            gaze_ratio = (
                left_ratio +
                right_ratio
            ) / 2.0

            if gaze_ratio < 0.38:

                return "RIGHT"

            elif gaze_ratio > 0.62:

                return "LEFT"

            else:

                return "CENTER"

        except Exception:

            return "CENTER"

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    def process(
        self,
        frame
    ):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=
                mp.ImageFormat.SRGB,
            data=rgb
        )

        self.timestamp_ms += 33

        result = (
            self.landmarker
            .detect_for_video(
                mp_image,
                self.timestamp_ms
            )
        )

        if not result.face_landmarks:

            return None

        landmarks = (
            result.face_landmarks[0]
        )

        h, w = frame.shape[:2]

        xs = [
            landmark.x
            for landmark in
            landmarks
        ]

        ys = [
            landmark.y
            for landmark in
            landmarks
        ]

        min_x = max(
            0,
            int(min(xs) * w)
        )

        max_x = min(
            w - 1,
            int(max(xs) * w)
        )

        min_y = max(
            0,
            int(min(ys) * h)
        )

        max_y = min(
            h - 1,
            int(max(ys) * h)
        )

        face_x = int(
            (min_x + max_x) / 2
        )

        face_y = int(
            (min_y + max_y) / 2
        )

        horizontal, vertical = (
            self.calculate_direction(
                face_x,
                face_y,
                w,
                h
            )
        )

        tilt_angle = (
            self.calculate_tilt(
                landmarks
            )
        )

        if (
            tilt_angle <
            -TILT_DEADZONE
        ):

            tilt_direction = "RIGHT"

        elif (
            tilt_angle >
            TILT_DEADZONE
        ):

            tilt_direction = "LEFT"

        else:

            tilt_direction = "CENTER"

        gaze = (
            self.calculate_gaze(
                landmarks
            )
        )

        return {

            "box": (
                min_x,
                min_y,
                max_x,
                max_y
            ),

            "center": (
                face_x,
                face_y
            ),

            "horizontal":
                horizontal,

            "vertical":
                vertical,

            "tilt_angle":
                tilt_angle,

            "tilt_direction":
                tilt_direction,

            "gaze":
                gaze,

            "landmarks":
                landmarks
        }


# ============================================================
# ============================================================
# YOLO PROCESSOR
# ============================================================
# ============================================================

class YOLOProcessor:

    def __init__(self):

        print(
            "Loading YOLO model..."
        )

        self.model = YOLO(
            YOLO_MODEL_PATH
        )

        print(
            "YOLO model loaded."
        )

    def process(
        self,
        frame
    ):

        results = (
            self.model.predict(

                source=frame,

                conf=
                    YOLO_CONFIDENCE,

                verbose=False,

                device="cpu",

                imgsz=320
            )
        )

        if not results:

            return [], []

        result = results[0]

        persons = []

        objects = []

        if result.boxes is None:

            return (
                persons,
                objects
            )

        names = result.names

        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )

            class_id = int(
                box.cls[0]
            )

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            class_name = (
                names[class_id]
            )

            detection = {

                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                ),

                "confidence":
                    confidence,

                "class_name":
                    class_name
            }

            if (
                class_name.lower()
                ==
                "person"
            ):

                persons.append({

                    "box": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),

                    "confidence":
                        confidence
                })

            else:

                objects.append(
                    detection
                )

        return (
            persons,
            objects
        )


# ============================================================
# ============================================================
# LOAD KNOWN FACES
# ============================================================
# ============================================================

def load_known_faces():

    with known_faces_lock:

        known_faces.clear()

        if not os.path.exists(
            KNOWN_FACES_DIR
        ):

            return

        for filename in os.listdir(
            KNOWN_FACES_DIR
        ):

            if not filename.lower().endswith(
                ".npz"
            ):

                continue

            path = os.path.join(
                KNOWN_FACES_DIR,
                filename
            )

            try:

                data = np.load(
                    path
                )

                if "embedding" not in data:

                    continue

                embedding = (
                    data["embedding"]
                    .astype(
                        np.float32
                    )
                )

                norm = np.linalg.norm(
                    embedding
                )

                if norm == 0:

                    continue

                embedding = (
                    embedding /
                    norm
                )

                name = os.path.splitext(
                    filename
                )[0]

                known_faces[name] = (
                    embedding
                )

                print(
                    f"Loaded face: {name}"
                )

            except Exception as e:

                print(
                    f"Could not load "
                    f"{filename}: {e}"
                )

    print(
        f"Known people: "
        f"{len(known_faces)}"
    )


# ============================================================
# FACE RECOGNITION
# ============================================================

def recognize_embedding(
    embedding
):

    if embedding is None:

        return (
            "UNKNOWN",
            0.0
        )

    embedding = (
        embedding.astype(
            np.float32
        )
    )

    norm = np.linalg.norm(
        embedding
    )

    if norm == 0:

        return (
            "UNKNOWN",
            0.0
        )

    embedding = (
        embedding / norm
    )

    best_name = "UNKNOWN"

    best_similarity = -1.0

    with known_faces_lock:

        for (
            name,
            known_embedding
        ) in known_faces.items():

            similarity = float(
                np.dot(
                    embedding,
                    known_embedding
                )
            )

            if (
                similarity >
                best_similarity
            ):

                best_similarity = (
                    similarity
                )

                best_name = name

    if (
        best_similarity
        >=
        RECOGNITION_THRESHOLD
    ):

        return (
            best_name,
            best_similarity
        )

    return (
        "UNKNOWN",
        best_similarity
    )


# ============================================================
# INSIGHTFACE
# ============================================================

print()

print(
    "Loading InsightFace..."
)

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=[
        "CPUExecutionProvider"
    ]
)

face_app.prepare(
    ctx_id=0,
    det_size=(320, 320)
)

print(
    "InsightFace loaded."
)


# ============================================================
# BACKGROUND INSIGHTFACE RECOGNITION
# ============================================================

def face_recognition_thread():

    global current_person
    global current_similarity

    frame_counter = 0

    print(
        "Face recognition thread started."
    )

    while running:

        if enrollment_active:

            time.sleep(
                0.1
            )

            continue

        with camera_lock:

            if latest_frame is None:

                frame = None

            else:

                frame = (
                    latest_frame.copy()
                )

        if frame is None:

            time.sleep(
                0.05
            )

            continue

        frame_counter += 1

        # Use a lighter recognition rate while voice is active.
        if voice_processing:
            current_recognition_interval = (
                FACE_RECOGNITION_INTERVAL_VOICE
            )
        else:
            current_recognition_interval = (
                FACE_RECOGNITION_INTERVAL
            )

        if (
            frame_counter %
            current_recognition_interval
            != 0
        ):

            time.sleep(
                0.005
            )

            continue

        try:

            faces = face_app.get(
                frame
            )

        except Exception as e:

            print(
                "Face recognition error:",
                e
            )

            continue

        recognized_name = "UNKNOWN"

        recognized_similarity = 0.0

        if faces:

            main_face = max(
                faces,
                key=lambda f:
                (
                    f.bbox[2] -
                    f.bbox[0]
                )
                *
                (
                    f.bbox[3] -
                    f.bbox[1]
                )
            )

            (
                recognized_name,
                recognized_similarity
            ) = recognize_embedding(
                main_face.embedding
            )

        with current_person_lock:

            current_person = (
                recognized_name
            )

            current_similarity = (
                recognized_similarity
            )

    print(
        "Face recognition thread stopped."
    )


# ============================================================
# CV BACKGROUND THREAD
# ============================================================

def cv_processing_thread():

    global cv_people
    global cv_targets
    global cv_objects
    global cv_face_data
    global cv_face_timestamp
    global cv_selected_target
    global cv_fps

    print(
        "CV processing thread started."
    )

    target_tracker = (
        StableTargetTracker()
    )

    face_processor = (
        FaceProcessor()
    )

    yolo = YOLOProcessor()

    frame_counter = 0

    last_face_result = None

    fps_counter = 0

    fps_start = time.time()

    try:

        while running:

            with camera_lock:

                if latest_frame is None:

                    frame = None

                else:

                    frame = (
                        latest_frame.copy()
                    )

            if frame is None:

                time.sleep(
                    0.05
                )

                continue

            frame_counter += 1

            # Use lighter CV processing while voice is active.
            # Previous valid CV results remain available to AI.

            if voice_processing:
                current_yolo_interval = (
                    YOLO_INTERVAL_VOICE
                )
                current_face_interval = (
                    FACE_INTERVAL_VOICE
                )
            else:
                current_yolo_interval = (
                    YOLO_INTERVAL
                )
                current_face_interval = (
                    FACE_INTERVAL
                )

            # =================================================
            # YOLO
            # =================================================

            if (
                frame_counter %
                current_yolo_interval
                == 0
            ):

                try:

                    people, objects = (
                        yolo.process(
                            frame
                        )
                    )

                    targets = (
                        target_tracker.update(
                            people
                        )
                    )

                    selected_target = None

                    if targets:

                        selected_target = max(
                            targets,
                            key=lambda target:
                            (
                                target["box"][2]
                                -
                                target["box"][0]
                            )
                            *
                            (
                                target["box"][3]
                                -
                                target["box"][1]
                            )
                        )

                    with cv_state_lock:

                        cv_people = people

                        cv_targets = targets

                        cv_objects = objects

                        cv_selected_target = (
                            selected_target
                        )

                except Exception as e:

                    print(
                        "YOLO error:",
                        e
                    )

            # =================================================
            # MEDIAPIPE FACE
            # =================================================

            if (
                frame_counter %
                current_face_interval
                == 0
            ):

                try:

                    result = (
                        face_processor.process(
                            frame
                        )
                    )

                    if result is not None:

                        last_face_result = (
                            result
                        )

                        with cv_state_lock:

                            cv_face_data = (
                                result
                            )

                            cv_face_timestamp = (
                                time.monotonic()
                            )

                except Exception as e:

                    print(
                        "MediaPipe error:",
                        e
                    )

            # =================================================
            # FPS
            # =================================================

            fps_counter += 1

            current_time = time.time()

            elapsed = (
                current_time -
                fps_start
            )

            if elapsed >= 1.0:

                fps = (
                    fps_counter /
                    elapsed
                )

                fps_counter = 0

                fps_start = (
                    current_time
                )

                with cv_state_lock:

                    cv_fps = fps

            time.sleep(
                0.001
            )

    finally:

        face_processor.landmarker.close()

        print(
            "CV processing thread stopped."
        )


# ============================================================
# ============================================================
# CAMERA → SERVO CONTROL
# ============================================================
# ============================================================

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def move_toward(current, target, max_step):
    """Move toward target without overshooting, one step-limited tick at a time."""
    delta = target - current
    if abs(delta) <= max_step:
        return target
    return current + max_step if delta > 0 else current - max_step


def face_to_eye_horizontal(face_x, frame_width):
    """Map stable image X to calibrated horizontal eye angle."""
    x = clamp(face_x / float(frame_width), 0.0, 1.0)
    error = x - 0.5
    center_zone = 0.06

    if abs(error) <= center_zone:
        return EYE_H_CENTER

    if error > 0:
        usable = (error - center_zone) / (0.5 - center_zone)
    else:
        usable = (error + center_zone) / (0.5 - center_zone)

    return clamp(
        EYE_H_CENTER + usable * 45.0,
        EYE_H_MIN,
        EYE_H_MAX
    )


def face_to_eye_vertical(face_y, frame_height):
    """Map stable image Y to calibrated vertical eye angle."""
    y = clamp(face_y / float(frame_height), 0.0, 1.0)
    error = y - 0.5
    center_zone = 0.10

    if abs(error) <= center_zone:
        return EYE_V_CENTER

    if error > 0:
        usable = (error - center_zone) / (0.5 - center_zone)
    else:
        usable = (error + center_zone) / (0.5 - center_zone)

    return clamp(
        EYE_V_CENTER + usable * 45.0,
        EYE_V_MIN,
        EYE_V_MAX
    )


def camera_servo_control_thread():
    """
    Hierarchical, symmetric eyes -> neck -> eyes face tracking controller.

    Tracking sequence:
        1. Eyes follow the face on both axes, but a new target is only
           accepted after it clears a hysteresis band AND persists for
           several consecutive frames (EYE_CONFIRM_FRAMES). This is what
           stops single-frame detector noise from producing the
           left<->right two-point oscillation.
        2. When the horizontal error grows large enough on EITHER side
           (symmetric bands, not a one-sided check), the neck activates.
        3. At the moment of hand-off (a single discrete event, not every
           tick) the horizontal eye target is set to center and held
           there only while the neck is actively engaged.
        4. The neck moves in small incremental steps, at a reduced
           command rate, with its speed slewed rather than jumping --
           this removes the overshoot "whip" past the face.
        5. Both eye axes are held (frozen) during neck motion and for a
           short cool-down window afterward, because neck rotation
           itself shifts the face's apparent position in frame (the
           camera is mounted on the moving head).
        6. Once the face returns inside the inner release band for
           enough consecutive frames, the neck disengages and the eyes
           re-acquire the current face position once.

    Jaw remains completely independent and audio controlled.
    """
    global current_eye_horizontal
    global current_eye_vertical
    global current_neck
    global servo_control_running

    # ------------------------------------------------------------
    # Filtered face position
    # ------------------------------------------------------------
    filtered_x = None
    filtered_y = None

    # Eye targets only change when a candidate has been confirmed.
    stable_eye_target_h = EYE_H_CENTER
    stable_eye_target_v = EYE_V_CENTER

    # Candidate counters stop one/two-frame detector noise from
    # changing the accepted target.
    h_candidate = None
    h_candidate_count = 0
    v_candidate = None
    v_candidate_count = 0

    # ------------------------------------------------------------
    # Neck state machine (symmetric for both directions)
    # ------------------------------------------------------------
    neck_active = False
    neck_direction = 0          # +1 or -1, set once at hand-off
    neck_start_count = 0
    neck_release_count = 0

    # Tracks the previous tick's compensated error so neck motion can
    # apply derivative braking (slow down when the gap is already
    # closing quickly, instead of driving in at full speed).
    previous_compensated_error = None
    neck_last_motion_time = time.monotonic()

    last_sent_h = None
    last_sent_v = None
    last_sent_neck = None
    previous_time = time.monotonic()

    print(
        "Camera servo-control thread started "
        "(symmetric hierarchical eyes -> neck -> eyes tracking)."
    )

    while running and servo_control_running:
        loop_start = time.monotonic()

        try:
            now = time.monotonic()

            dt = clamp(
                now - previous_time,
                0.001,
                0.10
            )
            previous_time = now

            with cv_state_lock:
                face_data = cv_face_data
                face_timestamp = cv_face_timestamp

            have_data = (
                face_data is not None
                and face_timestamp > 0
            )
            data_age = (now - face_timestamp) if have_data else None

            # "Fresh": recent enough to actively steer from this tick.
            data_fresh = (
                have_data and data_age <= FACE_STALE_TIMEOUT
            )

            # "Confirmed lost": no update for a genuinely long stretch --
            # only NOW do we release the neck and recenter the eyes.
            confirmed_lost = (
                not have_data or data_age > FACE_LOST_RESET_TIMEOUT
            )

            # In between fresh and confirmed_lost the data is merely
            # stale (a brief CV/CPU hiccup). We deliberately do nothing
            # in that window -- see FACE STALE (HOLD) branch below.

            # Axes are held (frozen against new targets) while the neck
            # is active and for a short cool-down window afterward, since
            # neck rotation itself moves the face in-frame.
            neck_recently_moving_h = (
                now - neck_last_motion_time
                < NECK_HORIZONTAL_HOLD_SECONDS
            )
            neck_recently_moving_v = (
                now - neck_last_motion_time
                < NECK_VERTICAL_HOLD_SECONDS
            )

            # ========================================================
            # FACE DATA FRESH -- actively track
            # ========================================================
            if data_fresh:
                raw_x, raw_y = face_data["center"]
                raw_x = float(raw_x)
                raw_y = float(raw_y)

                if filtered_x is None:
                    filtered_x = raw_x
                    filtered_y = raw_y
                    stable_eye_target_h = face_to_eye_horizontal(
                        filtered_x, FRAME_WIDTH
                    )
                    stable_eye_target_v = face_to_eye_vertical(
                        filtered_y, FRAME_HEIGHT
                    )
                    h_candidate = None
                    h_candidate_count = 0
                    v_candidate = None
                    v_candidate_count = 0
                else:
                    filtered_x = (
                        FACE_EMA_ALPHA_X * raw_x
                        + (1.0 - FACE_EMA_ALPHA_X) * filtered_x
                    )
                    filtered_y = (
                        FACE_EMA_ALPHA_Y * raw_y
                        + (1.0 - FACE_EMA_ALPHA_Y) * filtered_y
                    )

                desired_h = face_to_eye_horizontal(
                    filtered_x, FRAME_WIDTH
                )
                desired_v = face_to_eye_vertical(
                    filtered_y, FRAME_HEIGHT
                )

                # ====================================================
                # HORIZONTAL EYE TARGET -- STABLE BUT RESPONSIVE
                # ====================================================
                # A new target is only accepted while the neck is NOT
                # engaged and NOT within its post-motion hold window,
                # and only after it clears the hysteresis band and
                # persists in the same direction for EYE_CONFIRM_FRAMES.
                if not neck_active and not neck_recently_moving_h:
                    h_error = desired_h - stable_eye_target_h

                    if abs(h_error) >= EYE_TARGET_HYSTERESIS_DEGREES:
                        if h_candidate is None:
                            h_candidate = desired_h
                            h_candidate_count = 1
                        else:
                            candidate_delta = desired_h - h_candidate
                            if candidate_delta * h_error >= 0:
                                h_candidate = desired_h
                                h_candidate_count += 1
                            else:
                                h_candidate = desired_h
                                h_candidate_count = 1

                        if h_candidate_count >= EYE_CONFIRM_FRAMES:
                            stable_eye_target_h = h_candidate
                            h_candidate = None
                            h_candidate_count = 0
                    else:
                        h_candidate = None
                        h_candidate_count = 0
                else:
                    h_candidate = None
                    h_candidate_count = 0

                # ====================================================
                # VERTICAL EYE TARGET
                # ====================================================
                if not neck_active and not neck_recently_moving_v:
                    v_error = desired_v - stable_eye_target_v

                    if abs(v_error) >= EYE_TARGET_V_HYSTERESIS_DEGREES:
                        if v_candidate is None:
                            v_candidate = desired_v
                            v_candidate_count = 1
                        else:
                            candidate_delta = desired_v - v_candidate
                            if candidate_delta * v_error >= 0:
                                v_candidate = desired_v
                                v_candidate_count += 1
                            else:
                                v_candidate = desired_v
                                v_candidate_count = 1

                        if v_candidate_count >= EYE_CONFIRM_FRAMES:
                            stable_eye_target_v = v_candidate
                            v_candidate = None
                            v_candidate_count = 0
                    else:
                        v_candidate = None
                        v_candidate_count = 0
                else:
                    v_candidate = None
                    v_candidate_count = 0

                # ========================================================
                # NECK HAND-OFF DECISION -- HEAD-RELATIVE, SYMMETRIC
                # ========================================================
                # raw_angle_deg is the face's offset from image-center
                # converted to the same degree scale as the eye servo
                # (so a full-frame edge maps to the eye's own +/-45
                # degree range). eye_deflection_deg is how far the eye
                # is CURRENTLY holding away from its center to look at
                # the face.
                #
                # compensated_error_deg = eye_deflection_deg + raw_angle_deg
                # is the face's TRUE angle relative to the head/neck,
                # independent of where the eye is currently pointed.
                # This is what fixes the "hand-off fires repeatedly but
                # the neck barely moves" bug: previously the decision
                # used raw_angle_deg alone, which collapses toward zero
                # the instant the eyes recenter (because the camera is
                # mounted in the eye and swings with it) -- making the
                # controller think the target was reached when really
                # only the eye moved, not the neck.
                normalized_x = (
                    clamp(filtered_x / float(FRAME_WIDTH), 0.0, 1.0)
                    - 0.5
                )
                eye_half_range = (EYE_H_MAX - EYE_H_CENTER)
                raw_angle_deg = normalized_x * 2.0 * eye_half_range

                eye_deflection_deg = (
                    current_eye_horizontal - EYE_H_CENTER
                )

                compensated_error_deg = (
                    eye_deflection_deg + raw_angle_deg
                )
                abs_compensated_error = abs(compensated_error_deg)

                if not neck_active:
                    if abs_compensated_error >= NECK_START_DEG:
                        neck_start_count += 1
                    else:
                        neck_start_count = 0

                    if neck_start_count >= NECK_CONFIRM_FRAMES:
                        neck_active = True
                        neck_start_count = 0
                        neck_release_count = 0
                        previous_compensated_error = None

                        # Direction is captured ONCE, at hand-off, from
                        # the sign of the compensated error -- it is not
                        # re-read every tick while the neck is moving.
                        neck_direction = (
                            1 if compensated_error_deg > 0 else -1
                        ) * NECK_IMAGE_RIGHT_SIGN
                        neck_direction = 1 if neck_direction > 0 else -1

                        # CRITICAL HAND-OFF (single event, not per-tick):
                        # once the neck takes over, stop asking the eyes
                        # to chase the side position -- they become a
                        # center-seeking stage while the neck follows,
                        # at EYE_RECENTER_SPEED (paced with the neck, not
                        # the fast normal tracking speed).
                        stable_eye_target_h = EYE_H_CENTER
                        h_candidate = None
                        h_candidate_count = 0

                        print(
                            "[SERVO] Neck hand-off ACTIVE "
                            f"(direction={neck_direction}, "
                            f"compensated_error={compensated_error_deg:.1f} deg) "
                            "-> eyes recenter"
                        )

                else:
                    # Release only after the compensated (head-relative)
                    # angle has returned well inside the inner band for
                    # several consecutive frames -- hysteresis prevents
                    # engage/release flicker right at the boundary, and
                    # using the compensated angle (not raw image
                    # position) means the eyes recentering can no longer
                    # trigger a false release on its own.
                    if abs_compensated_error <= NECK_STOP_DEG:
                        neck_release_count += 1
                    else:
                        neck_release_count = 0

                    if neck_release_count >= NECK_RELEASE_FRAMES:
                        neck_active = False
                        neck_direction = 0
                        neck_release_count = 0
                        neck_start_count = 0
                        previous_compensated_error = None
                        h_candidate = None
                        h_candidate_count = 0

                        # Re-acquire the eye target once, from the
                        # current filtered face position.
                        stable_eye_target_h = face_to_eye_horizontal(
                            filtered_x, FRAME_WIDTH
                        )

                        print(
                            "[SERVO] Neck hand-off RELEASED -> eyes resume"
                        )

                # ========================================================
                # NECK MOTION -- CONTINUOUS, EVERY TICK, SYMMETRIC
                # ========================================================
                # Updated every servo loop tick (not throttled to a
                # slower fixed command interval) so the motion glides
                # instead of visibly stepping. Speed is proportional to
                # the remaining compensated error, then damped further
                # by how fast that error is ALREADY closing (derivative
                # braking) -- if the gap is shrinking quickly on its
                # own, ease off instead of driving in at full speed and
                # needing to correct back afterward.
                if (
                    neck_active
                    and neck_direction != 0
                    and abs_compensated_error > NECK_STOP_DEG
                ):
                    motion_error = clamp(
                        (abs_compensated_error - NECK_STOP_DEG)
                        / max(1e-6, (NECK_FULL_SPEED_DEG - NECK_STOP_DEG)),
                        0.0,
                        1.0
                    )

                    neck_speed = (
                        NECK_TRACK_SPEED_MIN
                        + motion_error
                        * (NECK_TRACK_SPEED_MAX - NECK_TRACK_SPEED_MIN)
                    )

                    if previous_compensated_error is not None:
                        error_rate = (
                            previous_compensated_error
                            - abs_compensated_error
                        ) / dt

                        if error_rate > NECK_ERROR_RATE_FAST_DEG:
                            neck_speed *= NECK_BRAKE_FAST_FACTOR
                        elif error_rate > NECK_ERROR_RATE_MED_DEG:
                            neck_speed *= NECK_BRAKE_MED_FACTOR

                    neck_speed = clamp(
                        neck_speed,
                        NECK_TRACK_SPEED_MIN,
                        NECK_TRACK_SPEED_MAX
                    )

                    neck_step = neck_speed * dt

                    with servo_lock:
                        old_neck = current_neck
                        current_neck = clamp(
                            current_neck + neck_step * neck_direction,
                            NECK_MIN,
                            NECK_MAX
                        )

                    if abs(current_neck - old_neck) > 0.001:
                        neck_last_motion_time = now

                        if current_neck >= NECK_MAX:
                            print(
                                "[SERVO] Neck reached maximum "
                                "safe angle."
                            )
                        elif current_neck <= NECK_MIN:
                            print(
                                "[SERVO] Neck reached minimum "
                                "safe angle."
                            )

                previous_compensated_error = abs_compensated_error

            # ============================================================
            # FACE DATA NOT FRESH -- either a brief hiccup (HOLD) or a
            # genuinely lost face (RESET)
            # ============================================================
            elif confirmed_lost:
                # FACE CONFIRMED LOST: no update for FACE_LOST_RESET_TIMEOUT.
                # Only now do we actually release the neck and recenter.
                filtered_x = None
                filtered_y = None
                previous_compensated_error = None

                h_candidate = None
                h_candidate_count = 0
                v_candidate = None
                v_candidate_count = 0

                neck_active = False
                neck_direction = 0
                neck_start_count = 0
                neck_release_count = 0

                stable_eye_target_h = EYE_H_CENTER
                stable_eye_target_v = EYE_V_CENTER

            else:
                # FACE STALE BUT NOT YET LOST: a transient CV/CPU gap
                # (very common while STT/LLM/TTS are running on the same
                # machine). Deliberately do nothing here -- keep every
                # filter, target, and neck state exactly as it was. The
                # servo movement step below will simply keep commanding
                # the same target it already had, so nothing visibly
                # moves or resets. This is what stops the "hand-off
                # fires repeatedly but the neck barely moves" pattern.
                pass

            # ============================================================
            # PHYSICAL SERVO MOVEMENT
            # ============================================================
            # dt was already computed once at the top of this tick.

            # Horizontal eyes use a slower, dedicated recenter speed
            # while the neck is actively engaged, paced closer to the
            # neck's own closing speed -- this is what makes the eye
            # return and the neck rotation read as one simultaneous
            # motion instead of the eyes visibly finishing first.
            eye_step_h = (
                EYE_RECENTER_SPEED if neck_active else EYE_MAX_SPEED
            ) * dt
            eye_step_v = EYE_MAX_SPEED * dt
            neck_return_step = NECK_RETURN_SPEED * dt

            with servo_lock:
                current_eye_horizontal = move_toward(
                    current_eye_horizontal,
                    stable_eye_target_h,
                    eye_step_h
                )

                current_eye_vertical = move_toward(
                    current_eye_vertical,
                    stable_eye_target_v,
                    eye_step_v
                )

                # Return the neck only when the face is CONFIRMED lost
                # (not merely stale). A brief CV/CPU hiccup must never
                # cause an automatic center reset.
                if confirmed_lost:
                    current_neck = move_toward(
                        current_neck,
                        NECK_CENTER,
                        neck_return_step
                    )

                send_h = int(round(clamp(
                    current_eye_horizontal,
                    EYE_H_MIN,
                    EYE_H_MAX
                )))

                send_v = int(round(clamp(
                    current_eye_vertical,
                    EYE_V_MIN,
                    EYE_V_MAX
                )))

                send_neck = int(round(clamp(
                    current_neck,
                    NECK_MIN,
                    NECK_MAX
                )))

            # Publish only when the integer command changes.
            if send_h != last_sent_h:
                hardware_control.send_eye_horizontal(send_h)
                last_sent_h = send_h

            if send_v != last_sent_v:
                hardware_control.send_eye_vertical(send_v)
                last_sent_v = send_v

            if send_neck != last_sent_neck:
                hardware_control.send_neck(send_neck)
                last_sent_neck = send_neck

        except Exception as e:
            print("[SERVO] Camera servo-control error:", e)

        elapsed = time.monotonic() - loop_start
        time.sleep(
            max(
                0.005,
                SERVO_CONTROL_INTERVAL - elapsed
            )
        )

    print("Camera servo-control thread stopped.")


# ============================================================
# ============================================================
# GET CV STATE FOR AI
# ============================================================

def get_cv_context():

    with cv_state_lock:

        people_count = len(
            cv_people
        )

        selected_target = (
            cv_selected_target
        )

        face_data = cv_face_data

        objects = list(
            cv_objects
        )

    context = (
        f"People detected by camera: "
        f"{people_count}."
    )

    # --------------------------------------------------------
    # Person position
    # --------------------------------------------------------

    if face_data is not None:

        horizontal = (
            face_data["horizontal"]
        )

        vertical = (
            face_data["vertical"]
        )

        tilt = (
            face_data["tilt_direction"]
        )

        gaze = (
            face_data["gaze"]
        )

        context += (
            "\nCurrent face position: "
            f"{horizontal} horizontally, "
            f"{vertical} vertically."
        )

        context += (
            f"\nHead tilt: {tilt}."
        )

        context += (
            f"\nGaze direction: {gaze}."
        )

        # Human-readable position

        if (
            horizontal == "LEFT"
        ):

            position_text = (
                "The person is standing "
                "to the robot's left."
            )

        elif (
            horizontal == "RIGHT"
        ):

            position_text = (
                "The person is standing "
                "to the robot's right."
            )

        else:

            position_text = (
                "The person is standing "
                "approximately in front of "
                "the robot."
            )

        context += (
            "\nPOSITION INFORMATION: "
            + position_text
        )

    else:

        context += (
            "\nFace position is currently "
            "not available."
        )

    # --------------------------------------------------------
    # Closest person
    # --------------------------------------------------------

    if selected_target is not None:

        target_id = (
            selected_target["id"]
        )

        context += (
            f"\nClosest detected person "
            f"target ID: {target_id}."
        )

    # --------------------------------------------------------
    # Objects
    # --------------------------------------------------------

    if objects:

        object_names = []

        for obj in objects:

            object_names.append(
                obj["class_name"]
            )

        context += (
            "\nVisible objects: "
            +
            ", ".join(
                object_names
            )
        )

    return context


# ============================================================
# CAMERA DRAWING
# ============================================================

def draw_camera():

    with camera_lock:

        if latest_frame is None:

            return None

        frame = (
            latest_frame.copy()
        )

    with cv_state_lock:

        people = list(
            cv_people
        )

        targets = list(
            cv_targets
        )

        objects = list(
            cv_objects
        )

        face_data = (
            cv_face_data
        )

        selected_target = (
            cv_selected_target
        )

        fps = cv_fps

    with current_person_lock:

        name = current_person

        similarity = current_similarity

    # --------------------------------------------------------
    # People
    # --------------------------------------------------------

    for target in targets:

        x1, y1, x2, y2 = (
            target["box"]
        )

        target_id = (
            target["id"]
        )

        confidence = (
            target["confidence"]
        )

        color = (
            (0, 255, 255)
            if
            selected_target is not None
            and
            target_id ==
            selected_target["id"]
            else
            (0, 255, 0)
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            (
                f"PERSON ID:{target_id} "
                f"{confidence * 100:.0f}%"
            ),
            (
                x1,
                max(
                    20,
                    y1 - 8
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

    # --------------------------------------------------------
    # Objects
    # --------------------------------------------------------

    for obj in objects:

        x1, y1, x2, y2 = (
            obj["box"]
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 120, 0),
            2
        )

        cv2.putText(
            frame,
            (
                f"{obj['class_name']} "
                f"{obj['confidence'] * 100:.0f}%"
            ),
            (
                x1,
                max(
                    20,
                    y1 - 5
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 120, 0),
            2
        )

    # --------------------------------------------------------
    # Recognized identity
    # --------------------------------------------------------

    if name == "UNKNOWN":

        identity_text = (
            f"UNKNOWN "
            f"{max(similarity, 0) * 100:.1f}%"
        )

        identity_color = (
            0,
            165,
            255
        )

    else:

        identity_text = (
            f"{name} "
            f"{similarity * 100:.1f}%"
        )

        identity_color = (
            0,
            255,
            0
        )

    cv2.putText(
        frame,
        identity_text,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        identity_color,
        2
    )

    # --------------------------------------------------------
    # Face information
    # --------------------------------------------------------

    if face_data is not None:

        x1, y1, x2, y2 = (
            face_data["box"]
        )

        cx, cy = (
            face_data["center"]
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.circle(
            frame,
            (cx, cy),
            5,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            (
                f"HEAD H: "
                f"{face_data['horizontal']}"
            ),
            (20, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            (
                f"HEAD V: "
                f"{face_data['vertical']}"
            ),
            (20, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            (
                f"TILT: "
                f"{face_data['tilt_direction']} "
                f"({face_data['tilt_angle']:+.1f})"
            ),
            (20, 330),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            (
                f"GAZE: "
                f"{face_data['gaze']}"
            ),
            (20, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "FACE: NOT DETECTED",
            (20, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    # --------------------------------------------------------
    # People count
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"PEOPLE: {len(people)}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    if selected_target is not None:

        cv2.putText(
            frame,
            (
                f"TARGET ID: "
                f"{selected_target['id']}"
            ),
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "TARGET: NONE",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (520, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "FINAL VOICE + CAMERA TEST",
        (20, FRAME_HEIGHT - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    return frame


# ============================================================
# ROBOT SPEECH
# ============================================================

def robot_speak(
    text
):

    if not text:

        return

    print()
    print(
        "MASTER:"
    )

    print(
        text
    )

    try:

        tts_start = time.perf_counter()

        text_to_speech.text_to_speech(
            text
        )

        tts_time = (
            time.perf_counter()
            - tts_start
        )

        print(
            f"[TIMING] TTS generation: "
            f"{tts_time:.2f} seconds"
        )

        jaw_start = time.perf_counter()

        jaw_audio_sync.play_speech_with_jaw()

        jaw_time = (
            time.perf_counter()
            - jaw_start
        )

        print(
            f"[TIMING] Audio + jaw: "
            f"{jaw_time:.2f} seconds"
        )

    except Exception as e:

        print(
            "Speech/Jaw error:",
            e
        )


# ============================================================
# TTS LIMIT
# ============================================================

def prepare_tts_text(
    text
):

    if not text:

        return (
            "I'm sorry, I couldn't "
            "get a clear answer."
        )

    text = text.strip()

    if len(text) <= MAX_TTS_CHARACTERS:

        return text

    shortened = (
        text[:MAX_TTS_CHARACTERS]
    )

    positions = [

        shortened.rfind("."),

        shortened.rfind("?"),

        shortened.rfind("!")
    ]

    last_stop = max(
        positions
    )

    if last_stop > 250:

        return shortened[
            :last_stop + 1
        ]

    return (
        shortened + "..."
    )


# ============================================================
# WAKE WORD
# ============================================================

def is_wake_word(
    text
):

    if not text:

        return False

    text = (
        text.lower()
        .strip()
    )

    for word in WAKE_WORDS:

        if word in text:

            return True

    return False


# ============================================================
# EXIT COMMAND
# ============================================================

def is_exit_command(
    text
):

    if not text:

        return False

    text = (
        text.lower()
        .strip()
    )

    for word in EXIT_WORDS:

        if word in text:

            return True

    return False


# ============================================================
# SAVE NAME EXTRACTION
# ============================================================

def extract_save_name(
    text
):

    if not text:

        return None

    text_lower = (
        text.lower()
        .strip()
    )

    patterns = [

        r"save\s+(?:this\s+)?face\s+as\s+(.+)",

        r"save\s+my\s+face\s+as\s+(.+)",

        r"save\s+me\s+as\s+(.+)",

        r"remember\s+me\s+as\s+(.+)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text_lower
        )

        if not match:

            continue

        name = (
            match.group(1)
            .strip()
        )

        name = re.sub(
            r"\b(please|okay|ok|thanks|thank you)\b.*$",
            "",
            name
        ).strip()

        if not name:

            return None

        if name in [
            "as",
            "the",
            "this",
            "face",
            "me"
        ]:

            return None

        return name

    return None


# ============================================================
# CLEAN NAME
# ============================================================

def clean_name(
    name
):

    name = name.strip()

    name = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        name
    )

    return name.strip(
        "_"
    ).lower()


# ============================================================
# CURRENT PERSON
# ============================================================

def get_current_person():

    with current_person_lock:

        return (
            current_person,
            current_similarity
        )


# ============================================================
# HISTORY FILE
# ============================================================

def get_person_history_file(
    name
):

    if (
        not name
        or
        name == "UNKNOWN"
    ):

        return None

    clean_person_name = (
        clean_name(name)
    )

    if not clean_person_name:

        return None

    return os.path.join(
        CONVERSATION_DIR,
        f"{clean_person_name}.json"
    )


# ============================================================
# LOAD HISTORY
# ============================================================

def load_person_history(
    name
):

    if (
        not name
        or
        name == "UNKNOWN"
    ):

        return []

    canonical_path = (
        get_person_history_file(
            name
        )
    )

    if (
        canonical_path
        and
        os.path.exists(
            canonical_path
        )
    ):

        try:

            with open(
                canonical_path,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(
                    file
                )

            if isinstance(
                data,
                dict
            ):

                return data.get(
                    "conversation",
                    []
                )

            if isinstance(
                data,
                list
            ):

                return data

        except Exception as e:

            print(
                f"Could not load "
                f"{canonical_path}: {e}"
            )

    return []


# ============================================================
# PERSON CONTEXT
# ============================================================

def get_person_context():

    name, similarity = (
        get_current_person()
    )

    if name == "UNKNOWN":

        return (
            "The camera currently "
            "does not recognize "
            "the person."
        )

    context = (
        f"The camera recognizes "
        f"the current person as "
        f"{name}."
    )

    history = (
        load_person_history(
            name
        )
    )

    if history:

        context += (
            f"\nRecent conversation "
            f"history with {name}:\n"
        )

        for item in history[
            -MAX_HISTORY_MESSAGES:
        ]:

            role = item.get(
                "role",
                ""
            )

            content = item.get(
                "content",
                ""
            )

            context += (
                f"{role}: "
                f"{content}\n"
            )

    return context


# ============================================================
# CAPTURE FACE
# ============================================================

def capture_face_embedding():

    global enrollment_active

    print()
    print(
        "Starting face capture..."
    )

    print(
        f"Capture duration: "
        f"{2.0} seconds"
    )

    embeddings = []

    enrollment_active = True

    try:

        start_time = time.time()

        while (
            time.time() -
            start_time
            <
            2.0
        ):

            with camera_lock:

                frame = (
                    latest_frame.copy()
                    if latest_frame is not None
                    else None
                )

            if frame is None:

                time.sleep(
                    0.05
                )

                continue

            try:

                faces = face_app.get(
                    frame
                )

            except Exception as e:

                print(
                    "Enrollment "
                    "face-processing error:",
                    e
                )

                time.sleep(
                    0.05
                )

                continue

            if not faces:

                time.sleep(
                    0.05
                )

                continue

            face = max(
                faces,
                key=lambda f:
                (
                    f.bbox[2] -
                    f.bbox[0]
                )
                *
                (
                    f.bbox[3] -
                    f.bbox[1]
                )
            )

            if face.embedding is not None:

                embeddings.append(
                    face.embedding
                    .astype(
                        np.float32
                    )
                )

            time.sleep(
                0.05
            )

    finally:

        enrollment_active = False

    if len(embeddings) < 3:

        print(
            f"Only "
            f"{len(embeddings)} "
            f"samples captured."
        )

        return None

    average_embedding = (
        np.mean(
            np.stack(
                embeddings
            ),
            axis=0
        )
    )

    norm = np.linalg.norm(
        average_embedding
    )

    if norm == 0:

        return None

    average_embedding = (
        average_embedding /
        norm
    )

    return (
        average_embedding
        .astype(
            np.float32
        )
    )


# ============================================================
# SAVE FACE
# ============================================================

def save_face(
    name,
    embedding
):

    path = os.path.join(
        KNOWN_FACES_DIR,
        f"{name}.npz"
    )

    np.savez(
        path,
        embedding=embedding
    )

    with known_faces_lock:

        known_faces[name] = (
            embedding
        )

    print(
        f"Face saved: {path}"
    )


# ============================================================
# SAVE CONVERSATION
# ============================================================

def save_conversation_history(
    name,
    session_history,
    language
):

    if (
        not name
        or
        name == "UNKNOWN"
    ):

        return

    if not session_history:

        return

    clean_person_name = (
        clean_name(name)
    )

    if not clean_person_name:

        return

    path = (
        get_person_history_file(
            clean_person_name
        )
    )

    if not path:

        return

    existing_history = (
        load_person_history(
            clean_person_name
        )
    )

    existing_history.extend(
        session_history
    )

    data = {

        "person_name":
            clean_person_name,

        "language":
            language,

        "saved_at":
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "conversation":
            existing_history
    }

    try:

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False
            )

        print(
            "Conversation history saved:"
        )

        print(
            path
        )

    except Exception as e:

        print(
            "History save error:",
            e
        )


# ============================================================
# ENROLL PERSON
# ============================================================

def enroll_person(
    name,
    session_history,
    language
):

    clean_person_name = (
        clean_name(name)
    )

    if not clean_person_name:

        robot_speak(
            "I could not understand "
            "the name."
        )

        return False

    message = (
        f"I am saving you as "
        f"{clean_person_name}. "
        f"Please stay still."
    )

    session_history.append({

        "role":
            "assistant",

        "content":
            message
    })

    robot_speak(
        message
    )

    embedding = (
        capture_face_embedding()
    )

    if embedding is None:

        robot_speak(
            "I could not capture "
            "your face clearly."
        )

        return False

    save_face(
        clean_person_name,
        embedding
    )

    confirmation = (
        f"Done. I will remember "
        f"you as "
        f"{clean_person_name}."
    )

    session_history.append({

        "role":
            "assistant",

        "content":
            confirmation
    })

    save_conversation_history(
        clean_person_name,
        session_history,
        language
    )

    robot_speak(
        confirmation
    )

    return True


# ============================================================
# ASK NAME
# ============================================================

def ask_for_save_name(
    session_history,
    language
):

    response = (
        "What name should I "
        "save you as?"
    )

    session_history.append({

        "role":
            "assistant",

        "content":
            response
    })

    robot_speak(
        response
    )

    speech_recognition.record_speech()

    result = (
        speech_recognition
        .speech_to_text()
    )

    if isinstance(
        result,
        tuple
    ):

        name_text = result[0]

    else:

        name_text = result

    if not name_text:

        return None

    return name_text.strip()


# ============================================================

# ============================================================
# SMART AI CONTEXT ROUTING
# ============================================================
# CV is calculated continuously, but CV information is only
# supplied to the LLM when the current question needs it.
# Simple CV questions are answered locally.
#
# IMPORTANT:
# Temporary CV/person context is NEVER placed into the
# persistent robot_conversation history by this file.
# ============================================================

def classify_cv_request(user_text):
    """
    Detect only CV-related intents. Supports English, Hindi and Marathi.
    """
    text = (user_text or "").lower().strip()
    fields = set()

    # ---------------- IDENTITY ----------------
    if any(re.search(p, text) for p in [
        r"\bwho am i\b",
        r"\bdo you recognize me\b",
        r"\bdo you know me\b",
        r"\bwhat(?:'s| is) my name\b",
        r"\bwho do you see\b",
        r"\bcan you recognize me\b",
        r"मैं कौन हूँ",
        r"मेरा नाम क्या है",
        r"क्या आप मुझे पहचानते",
        r"तुम मुझे पहचानते",
        r"मी कोण आहे",
        r"माझं नाव काय",
        r"तुम्ही मला ओळखता",
    ]):
        fields.add("identity")

    # ---------------- PEOPLE COUNT ----------------
    if (
        re.search(
            r"\b(how many|number of|count of)\b.*"
            r"\b(people|persons|person|humans|peoples)\b",
            text
        )
        or re.search(r"\b(people|persons|humans)\b.*\b(here|there|see|visible)\b", text)
        or re.search(r"(कितने|कितनी|कितना).*(लोग|व्यक्ति|इंसान)", text)
        or re.search(r"(किती|कितीजण|किती जण).*(लोक|व्यक्ती|माणसं|माणसे)", text)
    ):
        fields.add("people_count")

    # ---------------- HORIZONTAL ----------------
    if (
        (
            re.search(r"\b(left|right)\b", text)
            and re.search(
                r"\b(am i|i am|i'm|where am i|standing|sitting|position|side|located)\b",
                text
            )
        )
        or re.search(r"\b(which side|what side)\b", text)
        or re.search(r"(डावीकडे|उजवीकडे|डाव्या|उजव्या).*(कुठे|आहे|आहात)", text)
        or re.search(r"(बाईं|दाईं|बाजू).*(कहाँ|किधर|हो)", text)
    ):
        fields.add("horizontal")

    # ---------------- VERTICAL ----------------
    if (
        (
            re.search(r"\b(up|down|above|below|vertical|vertically)\b", text)
            and re.search(
                r"\b(am i|i am|where|position|standing|sitting|located)\b",
                text
            )
        )
        or re.search(r"\b(above or below|up or down)\b", text)
        or re.search(r"(वर|खाली|वरती|खालती|उंच|उंची).*(कुठे|आहे|आहात)", text)
        or re.search(r"(ऊपर|नीचे|ऊंचे|नीचे).*(कहाँ|किधर|हो)", text)
    ):
        fields.add("vertical")

    # ---------------- HEAD TILT ----------------
    if (
        re.search(r"\b(head|tilt|tilted|leaning)\b", text)
        or re.search(r"(डोकं|डोके).*(झुक|वाक)", text)
        or re.search(r"(सिर|सर).*(झुका|झुक)", text)
    ):
        fields.add("tilt")

    # ---------------- GAZE ----------------
    if (
        re.search(
            r"\b(look|looking|gaze|gazing|staring|eyes|eye)\b",
            text
        )
        or re.search(r"(कुठे|किधर).*(पाह|बघ|देख)", text)
        or re.search(r"(कहाँ|किधर).*(देख|देखते)", text)
    ):
        fields.add("gaze")

    # ---------------- TARGET ----------------
    if re.search(
        r"\b(closest|nearest|target|focusing|focused)\b", text
    ) or re.search(r"(सर्वात जवळ|जवळचा|लक्ष|फोकस)", text):
        fields.add("target")

    # ---------------- OBJECTS ----------------
    if re.search(
        r"\b(object|objects|visible|detect|detected|laptop|phone|bottle|chair|table|book|cup)\b",
        text
    ) or re.search(
        r"(वस्तू|लॅपटॉप|फोन|बाटली|खुर्ची|टेबल|पुस्तक|कप)", text
    ) or re.search(
        r"(वस्तु|लैपटॉप|फोन|बोतल|कुर्सी|टेबल|किताब|कप)", text
    ):
        fields.add("objects")

    if re.search(
        r"\b(what do you see|what can you see|what is in front of you|describe what you see)\b",
        text
    ) or re.search(
        r"(तुला काय दिसत|तुम्हें क्या दिखाई|आपको क्या दिख)", text
    ):
        fields.update({"people_count", "objects", "target"})

    return fields

def get_required_cv_context(fields):
    if not fields:
        return ""

    with cv_state_lock:
        people_count = len(cv_people)
        selected_target = cv_selected_target
        face_data = cv_face_data
        objects = list(cv_objects)

    parts = []

    if "people_count" in fields:
        parts.append(
            f"People detected: {people_count}"
        )

    if face_data is not None:

        if "horizontal" in fields:
            parts.append(
                f"Horizontal position: "
                f"{face_data['horizontal']}"
            )

        if "vertical" in fields:
            parts.append(
                f"Vertical position: "
                f"{face_data['vertical']}"
            )

        if "tilt" in fields:
            parts.append(
                f"Head tilt: "
                f"{face_data['tilt_direction']}"
            )

        if "gaze" in fields:
            parts.append(
                f"Gaze direction: "
                f"{face_data['gaze']}"
            )

    elif any(
        field in fields
        for field in (
            "horizontal",
            "vertical",
            "tilt",
            "gaze"
        )
    ):
        parts.append(
            "Requested face information "
            "is currently unavailable."
        )

    if "target" in fields:

        if selected_target is not None:
            parts.append(
                f"Closest person target ID: "
                f"{selected_target['id']}"
            )
        else:
            parts.append(
                "No closest person target "
                "is currently available."
            )

    if "objects" in fields:

        if objects:
            names = [
                obj["class_name"]
                for obj in objects
            ]

            parts.append(
                "Visible objects: "
                + ", ".join(names)
            )
        else:
            parts.append(
                "No objects are currently detected."
            )

    return "\n".join(parts)


def get_local_cv_answer(
    user_text,
    fields,
    language
):
    """Answer simple CV questions locally, without an LLM call."""

    if not fields:
        return None

    text = (user_text or "").lower()

    with cv_state_lock:
        people_count = len(cv_people)
        face_data = cv_face_data
        objects = list(cv_objects)

    # --------------------------------------------------------
    # Identity
    # --------------------------------------------------------
    if fields == {"identity"}:
        person_name, similarity = get_current_person()

        if not person_name or person_name == "UNKNOWN":
            if language == "Hindi":
                return "मैं अभी आपको पहचान नहीं पा रहा हूँ।"
            if language == "Marathi":
                return "मी सध्या तुम्हाला ओळखू शकत नाही."
            return "I don't recognize you right now."

        if language == "Hindi":
            return f"आप {person_name} हैं।"
        if language == "Marathi":
            return f"तुम्ही {person_name} आहात."
        return f"You are {person_name}."

    # --------------------------------------------------------
    # People count
    # --------------------------------------------------------
    if fields == {"people_count"}:
        if language == "Hindi":
            return f"मुझे अभी {people_count} व्यक्ति दिखाई दे रहे हैं।"
        if language == "Marathi":
            return f"मला सध्या {people_count} व्यक्ती दिसत आहेत."
        return (
            f"I can see {people_count} "
            f"{'person' if people_count == 1 else 'people'} right now."
        )

    # --------------------------------------------------------
    # Horizontal
    # --------------------------------------------------------
    if fields == {"horizontal"}:
        if face_data is None:
            if language == "Hindi":
                return "मैं अभी आपकी क्षैतिज स्थिति निर्धारित नहीं कर सकता।"
            if language == "Marathi":
                return "मी सध्या तुमची आडवी स्थिती ठरवू शकत नाही."
            return "I can't determine your horizontal position right now."

        direction = face_data["horizontal"]

        if language == "Hindi":
            return {
                "LEFT": "आप मेरी बाईं ओर हैं।",
                "RIGHT": "आप मेरी दाईं ओर हैं।",
                "CENTER": "आप लगभग मेरे सामने हैं।"
            }[direction]

        if language == "Marathi":
            return {
                "LEFT": "तुम्ही माझ्या डाव्या बाजूला आहात.",
                "RIGHT": "तुम्ही माझ्या उजव्या बाजूला आहात.",
                "CENTER": "तुम्ही माझ्यासमोर आहात."
            }[direction]

        if direction == "CENTER":
            return "You're approximately in front of me."
        return f"You're on my {direction.lower()}."

    # --------------------------------------------------------
    # Vertical
    # --------------------------------------------------------
    if fields == {"vertical"}:
        if face_data is None:
            if language == "Hindi":
                return "मैं अभी आपकी ऊर्ध्वाधर स्थिति निर्धारित नहीं कर सकता।"
            if language == "Marathi":
                return "मी सध्या तुमची उभी स्थिती ठरवू शकत नाही."
            return "I can't determine your vertical position right now."

        direction = face_data["vertical"]

        if language == "Hindi":
            if direction == "CENTER":
                return "आप लगभग मेरी समान ऊंचाई पर हैं।"
            return f"आप केंद्र से {('ऊपर' if direction == 'UP' else 'नीचे')} हैं।"

        if language == "Marathi":
            if direction == "CENTER":
                return "तुम्ही माझ्याइतक्याच उंचीवर आहात."
            return f"तुम्ही केंद्राच्या {('वर' if direction == 'UP' else 'खाली')} आहात."

        if direction == "CENTER":
            return "You're approximately at the same level as me."
        return f"You're {direction.lower()} from the center."

    # --------------------------------------------------------
    # Tilt
    # --------------------------------------------------------
    if fields == {"tilt"}:
        if face_data is None:
            if language == "Hindi":
                return "मैं अभी आपके सिर का झुकाव निर्धारित नहीं कर सकता।"
            if language == "Marathi":
                return "मी सध्या तुमच्या डोक्याचा कल ठरवू शकत नाही."
            return "I can't determine your head tilt right now."

        direction = face_data["tilt_direction"]

        if language == "Hindi":
            if direction == "CENTER":
                return "आपका सिर सीधा है।"
            return f"आपका सिर {direction.lower()} झुका हुआ है।"

        if language == "Marathi":
            if direction == "CENTER":
                return "तुमचे डोके सरळ आहे."
            return f"तुमचे डोके {direction.lower()} झुकले आहे."

        if direction == "CENTER":
            return "Your head is straight."
        return f"Your head is tilted {direction.lower()}."

    # --------------------------------------------------------
    # Gaze
    # --------------------------------------------------------
    if fields == {"gaze"}:
        if face_data is None:
            if language == "Hindi":
                return "मैं अभी यह निर्धारित नहीं कर सकता कि आप कहाँ देख रहे हैं।"
            if language == "Marathi":
                return "तुम्ही कुठे पाहत आहात हे मी सध्या ठरवू शकत नाही."
            return "I can't determine where you're looking right now."

        gaze = face_data["gaze"]

        if language == "Hindi":
            return f"आप {gaze.lower()} की ओर देख रहे हैं।"
        if language == "Marathi":
            return f"तुम्ही {gaze.lower()} दिशेला पाहत आहात."

        return f"You're looking {gaze.lower()}."

    # --------------------------------------------------------
    # Specific object presence / object list
    # --------------------------------------------------------
    if fields == {"objects"}:
        names = [obj["class_name"].lower() for obj in objects]

        requested = re.findall(
            r"\b(laptop|phone|bottle|chair|table|book|cup)\b",
            text
        )

        if requested:
            found = any(item in names for item in requested)

            if found:
                if language == "Hindi":
                    return "हाँ, मुझे वह दिखाई दे रहा है।"
                if language == "Marathi":
                    return "हो, मला ते दिसत आहे."
                return "Yes, I can see it."

            if language == "Hindi":
                return "नहीं, मुझे वह अभी दिखाई नहीं दे रहा है।"
            if language == "Marathi":
                return "नाही, मला ते सध्या दिसत नाही."
            return "No, I don't currently detect it."

        if names:
            if language == "Hindi":
                return "मुझे अभी " + ", ".join(names) + " दिखाई दे रहे हैं।"
            if language == "Marathi":
                return "मला सध्या " + ", ".join(names) + " दिसत आहेत."
            return "I can currently see " + ", ".join(names) + "."

        if language == "Hindi":
            return "मुझे अभी कोई वस्तु दिखाई नहीं दे रही है।"
        if language == "Marathi":
            return "मला सध्या कोणतीही वस्तू दिसत नाही."
        return "I don't currently detect any objects."

    return None

def build_smart_ai_input(
    user_text,
    language,
    person_context,
    fields
):
    """
    Build only the information required for THIS request.
    This string is temporary and must not be persisted.
    """

    parts = [
        f"Language: {language}"
    ]

    if "identity" in fields and person_context:
        parts.append(
            f"Person information:\n{person_context}"
        )

    cv_context = get_required_cv_context(fields)

    if cv_context:
        parts.append(
            "Camera information required "
            "for this question:\n"
            + cv_context
        )

    parts.append(
        "Answer concisely for spoken output."
    )

    parts.append(
        "Respond in exactly the same language "
        "as the user."
    )

    parts.append(
        "Do not invent visual information."
    )

    parts.append(
        f"User question:\n{user_text}"
    )

    return "\n\n".join(parts)


# ============================================================
# MAIN
# ============================================================
# ============================================================

def main():

    global running

    global voice_processing

    global enrollment_active

    load_known_faces()

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    camera = CameraStream(
        CAMERA_INDEX
    )

    # --------------------------------------------------------
    # Start CV
    # --------------------------------------------------------

    cv_thread = threading.Thread(
        target=cv_processing_thread,
        daemon=True
    )

    cv_thread.start()

    # --------------------------------------------------------
    # Face recognition
    # --------------------------------------------------------

    recognition_thread = threading.Thread(
        target=face_recognition_thread,
        daemon=True
    )

    recognition_thread.start()

    # --------------------------------------------------------
    # Camera → Eye/Neck servo control
    # --------------------------------------------------------

    # Connect the active MQTT controller.
    # jaw_audio_sync also uses hardware_control.
    hardware_control.connect()

    servo_thread = threading.Thread(
        target=camera_servo_control_thread,
        daemon=True
    )

    servo_thread.start()

    time.sleep(
        2.0
    )

    active = False

    session_history = []

    current_language = "English"

    current_session_person = (
        "UNKNOWN"
    )

    print()
    print("=" * 60)
    print(
        "FINAL VOICE + CAMERA "
        "INTEGRATION TEST"
    )
    print("=" * 60)

    print()

    print(
        "Wake word: Hey Master"
    )

    print(
        "Languages: "
        "English / Hindi / Marathi"
    )
    print()

    try:

        while True:

            # =================================================
            # CAMERA PREVIEW
            # =================================================
            # Camera display remains alive continuously.
            # Voice processing continues normally in parallel.

            frame = (
                draw_camera()
            )

            if frame is not None:

                cv2.imshow(
                    "Robot Head - "
                    "Final Integration",
                    frame
                )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if key == 27:

                break

            # =================================================
            # WAKE WORD
            # =================================================

            if not active:

                print()
                print(
                    "Waiting for wake word..."
                )

                speech_recognition.record_speech()

                stt_start = time.perf_counter()

                result = (
                    speech_recognition
                    .speech_to_text()
                )

                stt_time = (
                    time.perf_counter()
                    - stt_start
                )

                print(
                    f"[TIMING] STT: "
                    f"{stt_time:.2f} seconds"
                )

                if isinstance(
                    result,
                    tuple
                ):

                    wake_text = (
                        result[0]
                    )

                    wake_language = (
                        result[1]
                    )

                else:

                    wake_text = result

                    wake_language = (
                        "English"
                    )

                if is_wake_word(
                    wake_text
                ):

                    active = True

                    voice_processing = (
                        True
                    )

                    current_language = (
                        wake_language
                    )

                    session_history = []

                    current_session_person, _ = (
                        get_current_person()
                    )

                    greeting = (
                        "Yes, how can I "
                        "help you?"
                    )

                    session_history.append({

                        "role":
                            "assistant",

                        "content":
                            greeting
                    })

                    robot_speak(
                        greeting
                    )

                continue

            # =================================================
            # ACTIVE CONVERSATION
            # =================================================

            print()
            print(
                "Listening..."
            )

            speech_recognition.record_speech()

            stt_start = time.perf_counter()

            result = (
                speech_recognition
                .speech_to_text()
            )

            stt_time = (
                time.perf_counter()
                - stt_start
            )

            print(
                f"[TIMING] STT: "
                f"{stt_time:.2f} seconds"
            )

            if isinstance(
                result,
                tuple
            ):

                user_text = (
                    result[0]
                )

                language = (
                    result[1]
                )

            else:

                user_text = result

                language = (
                    "English"
                )

            if not user_text:

                continue

            user_text = (
                user_text.strip()
            )

            current_language = (
                language
            )

            print()
            print(
                "YOU:",
                user_text
            )

            print(
                "LANGUAGE:",
                language
            )

            # =================================================
            # EXIT
            # =================================================

            if is_exit_command(
                user_text
            ):

                person_name, _ = (
                    get_current_person()
                )

                if (
                    person_name
                    !=
                    "UNKNOWN"
                    and
                    session_history
                ):

                    save_conversation_history(
                        person_name,
                        session_history,
                        current_language
                    )

                active = False

                voice_processing = (
                    False
                )

                session_history = []

                print(
                    "Conversation ended."
                )

                continue

            # =================================================
            # SAVE FACE
            # =================================================

            save_name = (
                extract_save_name(
                    user_text
                )
            )

            if (
                save_name is None
                and
                re.search(
                    r"save.*face.*as",
                    user_text.lower()
                )
            ):

                session_history.append({

                    "role":
                        "user",

                    "content":
                        user_text
                })

                save_name = (
                    ask_for_save_name(
                        session_history,
                        language
                    )
                )

                if save_name:

                    session_history.append({

                        "role":
                            "user",

                        "content":
                            save_name
                    })

                    enroll_person(
                        save_name,
                        session_history,
                        language
                    )

                continue

            if save_name:

                session_history.append({

                    "role":
                        "user",

                    "content":
                        user_text
                })

                enroll_person(
                    save_name,
                    session_history,
                    language
                )

                continue

            # Ignore empty / punctuation-only STT results.
            # This prevents useless LLM calls and TTS errors.
            if not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿऀ-ॿ]", user_text or ""):
                print(
                    "[VOICE] Ignoring empty or punctuation-only speech."
                )
                continue

            if len((user_text or "").strip()) < 2:
                print(
                    "[VOICE] Ignoring extremely short speech."
                )
                continue

            # =================================================
            # NORMAL CONVERSATION
            # =================================================

            session_history.append({

                "role":
                    "user",

                "content":
                    user_text
            })

            # -------------------------------------------------
            # SMART CONTEXT ROUTING
            # -------------------------------------------------

            person_name, similarity = (
                get_current_person()
            )

            # -------------------------------------------------
            # ROBOT SELF-KNOWLEDGE ROUTING
            # -------------------------------------------------

            self_route = (
                robot_self_knowledge
                .classify_question(
                    user_text
                )
            )

            self_categories = (
                self_route.get(
                    "categories",
                    []
                )
            )

            self_knowledge = ""

            if self_route.get(
                "is_self",
                False
            ):
                self_knowledge = (
                    robot_self_knowledge
                    .get_knowledge(
                        self_categories
                    )
                )

            print(
                "[SELF ROUTER]",
                self_route
            )

            if self_knowledge:
                print(
                    "[SELF ROUTER] Categories:",
                    self_categories
                )

            fields = classify_cv_request(
                user_text
            )

            # Simple CV facts are answered locally.
            # This means no Groq tokens are consumed.
            local_answer = get_local_cv_answer(
                user_text,
                fields,
                language
            )

            if local_answer is not None:

                print(
                    "[SMART ROUTER] Local answer."
                )

                print(
                    f"[SMART ROUTER] Fields: "
                    f"{sorted(fields)}"
                )

                robot_text = local_answer

                session_history.append({
                    "role": "assistant",
                    "content": robot_text
                })

                spoken_text = prepare_tts_text(
                    robot_text
                )

                robot_speak(
                    spoken_text
                )

                continue

            # Only load person history when relevant.
            person_context = ""

            if (
                "identity" in fields
                or
                re.search(
                    r"\b(remember|discussed|last time|earlier|before|my name)\b",
                    user_text.lower()
                )
            ):
                person_context = get_person_context()

            cv_context = get_required_cv_context(fields)

            print(
                "[SMART ROUTER] LLM request."
            )

            print(
                f"[SMART ROUTER] CV fields sent: "
                f"{sorted(fields) if fields else 'NONE'}"
            )

            # =================================================
            # AI
            # =================================================

            ai_start = time.perf_counter()

            robot_text = (
                robot_conversation
                .get_robot_response(
                    user_text=user_text,
                    language=language,
                    cv_context=cv_context,
                    person_context=person_context,
                    robot_knowledge=self_knowledge
                )
            )

            ai_time = (
                time.perf_counter()
                - ai_start
            )

            print(
                f"[TIMING] AI response: "
                f"{ai_time:.2f} seconds"
            )

            session_history.append({

                "role":
                    "assistant",

                "content":
                    robot_text
            })

            spoken_text = (
                prepare_tts_text(
                    robot_text
                )
            )

            robot_speak(
                spoken_text
            )

            # =================================================
            # PERSON CHANGE
            # =================================================

            new_person, _ = (
                get_current_person()
            )

            if (
                current_session_person
                !=
                new_person
            ):

                if (
                    current_session_person
                    !=
                    "UNKNOWN"
                    and
                    session_history
                ):

                    save_conversation_history(
                        current_session_person,
                        session_history,
                        current_language
                    )

                current_session_person = (
                    new_person
                )

    except KeyboardInterrupt:

        print(
            "\nStopped by keyboard."
        )

    except Exception as e:

        print()
        print("=" * 60)
        print(
            "UNEXPECTED MAIN LOOP ERROR"
        )
        print("=" * 60)
        print(
            f"Error type: "
            f"{type(e).__name__}"
        )
        print(
            f"Error: {e}"
        )

        import traceback

        traceback.print_exc()

    finally:

        running = False

        servo_control_running = False

        voice_processing = False

        enrollment_active = False

        time.sleep(
            0.5
        )

        camera.stop()

        cv2.destroyAllWindows()

        try:
            hardware_control.disconnect()
        except Exception as e:
            print(
                "Hardware-control disconnect error:",
                e
            )

        print()
        print(
            "Final voice + camera "
            "integration stopped."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()