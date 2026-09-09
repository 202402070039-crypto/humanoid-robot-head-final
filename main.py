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
- Servo control is NOT included yet.
- MQTT is NOT included yet.
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
    "robot_head_camera",
    "models",
    "yolo11n.pt"
)

FACE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "robot_head_camera",
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
                    person_context=person_context
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

        voice_processing = False

        enrollment_active = False

        time.sleep(
            0.5
        )

        camera.stop()

        cv2.destroyAllWindows()

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