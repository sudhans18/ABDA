import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import math
import subprocess
import glob
import shutil
from collections import deque


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "face_landmarker.task"

# ------------------------------------------------------------
# MediaPipe / Facial features
# ------------------------------------------------------------

EAR_THRESHOLD = 0.21

GAZE_LEFT_THRESHOLD = 0.40
GAZE_RIGHT_THRESHOLD = 0.60

SMOOTHING_WINDOW = 5
GAZE_SMOOTHING_WINDOW = 5

# ------------------------------------------------------------
# Video recording
# ------------------------------------------------------------

RECORDED_VIDEO = "abda_session.mp4"

VIDEO_FPS = 30.0

# ------------------------------------------------------------
# OpenFace
# ------------------------------------------------------------

OPENFACE_EXE = (
    r"C:\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
)

# OpenFace output will be stored inside the project folder.
OPENFACE_OUTPUT_DIR = os.path.abspath(
    "openface_output"
)

AU_SUMMARY_FILE = "openface_au_summary.csv"

# ------------------------------------------------------------
# Output files
# ------------------------------------------------------------

CSV_FILE = "facial_session.csv"

SUMMARY_FILE = "facial_summary.txt"


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)

FaceLandmarker = (
    mp.tasks.vision.FaceLandmarker
)

FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)


if not os.path.exists(MODEL_PATH):

    print("ERROR: face_landmarker.task not found.")

    print("Make sure it is inside:")

    print(os.getcwd())

    raise SystemExit


options = FaceLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.VIDEO,

    num_faces=1,

    min_face_detection_confidence=0.5,

    min_face_presence_confidence=0.5,

    min_tracking_confidence=0.5,

    output_face_blendshapes=False,

    output_facial_transformation_matrixes=False

)


# ============================================================
# LANDMARK INDICES
# ============================================================

LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]

RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]


HEAD_POSE_POINTS = [
    1,
    152,
    33,
    263,
    61,
    291
]


LEFT_IRIS = [
    468,
    469,
    470,
    471,
    472
]

RIGHT_IRIS = [
    473,
    474,
    475,
    476,
    477
]


LEFT_EYE_CORNERS = [
    33,
    133
]

RIGHT_EYE_CORNERS = [
    362,
    263
]


# ============================================================
# BASIC DISTANCE
# ============================================================

def distance(p1, p2):

    return np.linalg.norm(
        np.array(p1) - np.array(p2)
    )


# ============================================================
# EAR
# ============================================================

def calculate_ear(
    landmarks,
    eye_indices
):

    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]

    vertical_1 = distance(p2, p6)

    vertical_2 = distance(p3, p5)

    horizontal = distance(p1, p4)

    if horizontal == 0:

        return 0.0

    ear = (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)

    return ear


# ============================================================
# GAZE
# ============================================================

def calculate_gaze(landmarks):

    # --------------------------------------------------------
    # LEFT EYE
    # --------------------------------------------------------

    left_iris_x = np.mean(
        [
            landmarks[i][0]
            for i in LEFT_IRIS
        ]
    )

    left_corner_1 = landmarks[
        LEFT_EYE_CORNERS[0]
    ][0]

    left_corner_2 = landmarks[
        LEFT_EYE_CORNERS[1]
    ][0]

    left_min_x = min(
        left_corner_1,
        left_corner_2
    )

    left_max_x = max(
        left_corner_1,
        left_corner_2
    )

    left_eye_width = (
        left_max_x - left_min_x
    )

    if left_eye_width <= 0:

        return "UNKNOWN", np.nan

    left_ratio = (
        left_iris_x - left_min_x
    ) / left_eye_width


    # --------------------------------------------------------
    # RIGHT EYE
    # --------------------------------------------------------

    right_iris_x = np.mean(
        [
            landmarks[i][0]
            for i in RIGHT_IRIS
        ]
    )

    right_corner_1 = landmarks[
        RIGHT_EYE_CORNERS[0]
    ][0]

    right_corner_2 = landmarks[
        RIGHT_EYE_CORNERS[1]
    ][0]

    right_min_x = min(
        right_corner_1,
        right_corner_2
    )

    right_max_x = max(
        right_corner_1,
        right_corner_2
    )

    right_eye_width = (
        right_max_x - right_min_x
    )

    if right_eye_width <= 0:

        return "UNKNOWN", np.nan

    right_ratio = (
        right_iris_x - right_min_x
    ) / right_eye_width


    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    gaze_ratio = (
        left_ratio + right_ratio
    ) / 2.0


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    if gaze_ratio < GAZE_LEFT_THRESHOLD:

        gaze_label = "LEFT"

    elif gaze_ratio > GAZE_RIGHT_THRESHOLD:

        gaze_label = "RIGHT"

    else:

        gaze_label = "CENTER"


    return gaze_label, gaze_ratio


# ============================================================
# HEAD POSE
# ============================================================

def calculate_head_pose(
    landmarks,
    width,
    height
):

    model_points = np.array([

        (0.0, 0.0, 0.0),

        (0.0, -330.0, -65.0),

        (-225.0, 170.0, -135.0),

        (225.0, 170.0, -135.0),

        (-150.0, -150.0, -125.0),

        (150.0, -150.0, -125.0)

    ], dtype=np.float64)


    image_points = np.array([

        (
            landmarks[1][0],
            landmarks[1][1]
        ),

        (
            landmarks[152][0],
            landmarks[152][1]
        ),

        (
            landmarks[33][0],
            landmarks[33][1]
        ),

        (
            landmarks[263][0],
            landmarks[263][1]
        ),

        (
            landmarks[61][0],
            landmarks[61][1]
        ),

        (
            landmarks[291][0],
            landmarks[291][1]
        )

    ], dtype=np.float64)


    # --------------------------------------------------------
    # Approximate camera matrix
    # --------------------------------------------------------

    focal_length = width

    camera_matrix = np.array([

        [
            focal_length,
            0,
            width / 2
        ],

        [
            0,
            focal_length,
            height / 2
        ],

        [
            0,
            0,
            1
        ]

    ], dtype=np.float64)


    distortion = np.zeros(
        (4, 1),
        dtype=np.float64
    )


    # --------------------------------------------------------
    # solvePnP
    # --------------------------------------------------------

    success, rotation_vector, translation_vector = (
        cv2.solvePnP(

            model_points,

            image_points,

            camera_matrix,

            distortion,

            flags=cv2.SOLVEPNP_ITERATIVE

        )
    )


    if not success:

        return None


    # --------------------------------------------------------
    # Rotation vector -> rotation matrix
    # --------------------------------------------------------

    rotation_matrix, _ = cv2.Rodrigues(
        rotation_vector
    )


    # --------------------------------------------------------
    # ZYX Euler angles
    # --------------------------------------------------------

    sy = math.sqrt(

        rotation_matrix[0, 0] ** 2
        +
        rotation_matrix[1, 0] ** 2

    )


    singular = sy < 1e-6


    if not singular:

        pitch = math.atan2(

            rotation_matrix[2, 1],

            rotation_matrix[2, 2]

        )

        yaw = math.atan2(

            -rotation_matrix[2, 0],

            sy

        )

        roll = math.atan2(

            rotation_matrix[1, 0],

            rotation_matrix[0, 0]

        )

    else:

        pitch = math.atan2(

            -rotation_matrix[1, 2],

            rotation_matrix[1, 1]

        )

        yaw = math.atan2(

            -rotation_matrix[2, 0],

            sy

        )

        roll = 0


    # --------------------------------------------------------
    # Radians -> degrees
    # --------------------------------------------------------

    pitch = math.degrees(pitch)

    yaw = math.degrees(yaw)

    roll = math.degrees(roll)


    return pitch, yaw, roll


# ============================================================
# SMOOTHING
# ============================================================

pitch_history = deque(
    maxlen=SMOOTHING_WINDOW
)

yaw_history = deque(
    maxlen=SMOOTHING_WINDOW
)

roll_history = deque(
    maxlen=SMOOTHING_WINDOW
)

gaze_history = deque(
    maxlen=GAZE_SMOOTHING_WINDOW
)


def smooth_value(
    history,
    value
):

    history.append(value)

    return (
        sum(history)
        /
        len(history)
    )


# ============================================================
# OPENFACE
# ============================================================

def run_openface(video_file):

    print("\n")
    print(
        "========================================"
    )

    print(
        "           OPENFACE AU ANALYSIS"
    )

    print(
        "========================================"
    )


    # --------------------------------------------------------
    # Check executable
    # --------------------------------------------------------

    if not os.path.exists(
        OPENFACE_EXE
    ):

        print(
            "\nOpenFace was not found."
        )

        print(
            "Expected path:"
        )

        print(
            OPENFACE_EXE
        )

        print(
            "\nAU analysis skipped."
        )

        return None


    # --------------------------------------------------------
    # Check video
    # --------------------------------------------------------

    if not os.path.exists(
        video_file
    ):

        print(
            "\nRecorded video not found:"
        )

        print(video_file)

        return None


    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OPENFACE_OUTPUT_DIR,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Remove old CSV files
    #
    # This prevents the program from accidentally
    # reading an old OpenFace result.
    # --------------------------------------------------------

    old_csv_files = glob.glob(
        os.path.join(
            OPENFACE_OUTPUT_DIR,
            "*.csv"
        )
    )


    for old_csv in old_csv_files:

        try:

            os.remove(old_csv)

        except Exception:

            pass


    # --------------------------------------------------------
    # OpenFace command
    # --------------------------------------------------------

    command = [

        OPENFACE_EXE,

        "-f",
        os.path.abspath(video_file),

        "-out_dir",
        OPENFACE_OUTPUT_DIR,

        "-aus",

        "-q"

    ]


    print(
        "\nRunning OpenFace..."
    )

    print(
        "This may take some time."
    )

    print(
        f"\nOpenFace output folder:"
    )

    print(
        OPENFACE_OUTPUT_DIR
    )


    try:

        result = subprocess.run(

            command,

            cwd=os.path.dirname(
                OPENFACE_EXE
            ),

            capture_output=True,

            text=True

        )

    except Exception as error:

        print(
            "\nCould not start OpenFace:"
        )

        print(error)

        return None


    # --------------------------------------------------------
    # Check return code
    # --------------------------------------------------------

    if result.returncode != 0:

        print(
            "\nOpenFace returned an error."
        )

        print(
            f"Return code: "
            f"{result.returncode}"
        )


        if result.stdout:

            print(
                "\nOpenFace output:"
            )

            print(result.stdout)


        if result.stderr:

            print(
                "\nOpenFace error:"
            )

            print(result.stderr)


        return None


    # --------------------------------------------------------
    # Find CSV
    # --------------------------------------------------------

    csv_files = glob.glob(

        os.path.join(

            OPENFACE_OUTPUT_DIR,

            "*.csv"

        )

    )


    if not csv_files:

        print(
            "\nOpenFace finished successfully,"
        )

        print(
            "but no CSV was found."
        )

        if result.stdout:

            print(
                "\nOpenFace output:"
            )

            print(result.stdout)

        return None


    # --------------------------------------------------------
    # Prefer the CSV matching the video name
    # --------------------------------------------------------

    video_name = os.path.splitext(
        os.path.basename(video_file)
    )[0]


    expected_csv = os.path.join(

        OPENFACE_OUTPUT_DIR,

        video_name + ".csv"

    )


    if os.path.exists(
        expected_csv
    ):

        csv_file = expected_csv

    else:

        csv_files.sort(

            key=os.path.getmtime,

            reverse=True

        )

        csv_file = csv_files[0]


    print(
        "\nOpenFace completed successfully."
    )

    print(
        "OpenFace CSV:"
    )

    print(
        csv_file
    )


    return csv_file


# ============================================================
# ANALYZE ACTION UNITS
# ============================================================

def analyze_action_units(
    csv_file
):

    if csv_file is None:
        return {}

    try:

        with open(
            csv_file,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            # OpenFace CSV headers contain leading spaces.
            # Example: " AU01_r"
            # Strip whitespace so AU columns are detected correctly.
            if reader.fieldnames:
                reader.fieldnames = [
                    column.strip() if column is not None else column
                    for column in reader.fieldnames
                ]

            rows = list(reader)

            # Clean row keys as well, for safety.
            cleaned_rows = []
            for row in rows:
                cleaned_row = {}
                for key, value in row.items():
                    if key is not None:
                        cleaned_row[key.strip()] = value
                cleaned_rows.append(cleaned_row)

            rows = cleaned_rows
            columns = reader.fieldnames or []

    except Exception as error:
        print("\nCould not read OpenFace CSV:")
        print(error)
        return {}

    if not columns:
        print("\nNo columns found in OpenFace CSV.")
        return {}

    # ========================================================
    # FIND AU COLUMNS
    # ========================================================

    intensity_columns = [
        column for column in columns
        if column and column.startswith("AU") and column.endswith("_r")
    ]

    presence_columns = [
        column for column in columns
        if column and column.startswith("AU") and column.endswith("_c")
    ]

    print(f"\nAU intensity columns found: {len(intensity_columns)}")
    print(f"AU presence columns found: {len(presence_columns)}")

    au_statistics = {}

    # ========================================================
    # AU INTENSITY
    # ========================================================

    for column in intensity_columns:
        values = []

        for row in rows:
            try:
                value = float(row[column])
                if np.isfinite(value):
                    values.append(value)
            except (ValueError, TypeError, KeyError):
                pass

        if len(values) == 0:
            continue

        values = np.array(values, dtype=float)

        au_statistics[column] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            # Percentage of valid frames with non-zero AU intensity.
            "activity_rate": float(np.mean(values > 0) * 100)
        }

    # ========================================================
    # AU PRESENCE
    # ========================================================

    for column in presence_columns:
        values = []

        for row in rows:
            try:
                value = float(row[column])
                if np.isfinite(value):
                    values.append(value)
            except (ValueError, TypeError, KeyError):
                pass

        if len(values) == 0:
            continue

        values = np.array(values, dtype=float)
        presence_rate = float(np.mean(values >= 1) * 100)

        # Example: AU12_c -> AU12_r
        au_number = column[:-2]
        intensity_name = au_number + "_r"

        if intensity_name in au_statistics:
            au_statistics[intensity_name]["presence_rate"] = presence_rate
        else:
            # Keep presence-only AUs such as AU28_c.
            au_statistics[column] = {
                "presence_rate": presence_rate
            }

    return au_statistics


# ============================================================
# SAVE AU SUMMARY
# ============================================================

def save_au_summary(
    au_statistics
):

    if not au_statistics:

        return


    try:

        with open(

            AU_SUMMARY_FILE,

            "w",

            newline="",

            encoding="utf-8"

        ) as file:

            writer = csv.writer(file)


            writer.writerow([

                "AU",

                "Mean",

                "Std",

                "Min",

                "Max",

                "ActivityRate",

                "PresenceRate"

            ])


            for au_name in sorted(
                au_statistics.keys()
            ):

                stats = (
                    au_statistics[au_name]
                )


                writer.writerow([

                    au_name,

                    stats.get(
                        "mean",
                        ""
                    ),

                    stats.get(
                        "std",
                        ""
                    ),

                    stats.get(
                        "min",
                        ""
                    ),

                    stats.get(
                        "max",
                        ""
                    ),

                    stats.get(
                        "activity_rate",
                        ""
                    ),

                    stats.get(
                        "presence_rate",
                        ""
                    )

                ])


        print(
            "\nSaved AU summary:"
        )

        print(
            f"  {AU_SUMMARY_FILE}"
        )


    except Exception as error:

        print(
            "\nCould not save AU summary:"
        )

        print(error)


# ============================================================
# PRINT AU SUMMARY
# ============================================================

def print_au_summary(
    au_statistics
):

    if not au_statistics:

        print(
            "\nNo Action Unit results available."
        )

        return


    print(
        "\nACTION UNITS"
    )

    print(
        "----------------------------------------"
    )


    # --------------------------------------------------------
    # Intensity
    # --------------------------------------------------------

    intensity_aus = [

        name

        for name in au_statistics

        if name.endswith("_r")

    ]


    if intensity_aus:

        print(
            "\nAU INTENSITY"
        )


        for au_name in sorted(
            intensity_aus
        ):

            stats = (
                au_statistics[au_name]
            )


            print(

                f"{au_name:<10} "

                f"Mean: "
                f"{stats['mean']:.2f}   "

                f"Std: "
                f"{stats['std']:.2f}   "

                f"Activity: "
                f"{stats['activity_rate']:.2f}%"

            )


    # --------------------------------------------------------
    # Presence
    # --------------------------------------------------------

    presence_aus = [

        name

        for name in au_statistics

        if name.endswith("_c")

    ]


    if presence_aus:

        print(
            "\nAU PRESENCE"
        )


        for au_name in sorted(
            presence_aus
        ):

            stats = (
                au_statistics[au_name]
            )


            print(

                f"{au_name:<10} "

                f"Presence: "
                f"{stats['presence_rate']:.2f}%"

            )


# ============================================================
# START WEBCAM
# ============================================================

print(
    "Opening webcam..."
)

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print(
        "ERROR: Could not open webcam."
    )

    raise SystemExit


# ============================================================
# CAMERA PROPERTIES
# ============================================================

frame_width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

frame_height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)


camera_fps = cap.get(
    cv2.CAP_PROP_FPS
)


if camera_fps <= 0:

    camera_fps = VIDEO_FPS


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)


video_writer = cv2.VideoWriter(

    RECORDED_VIDEO,

    fourcc,

    camera_fps,

    (
        frame_width,
        frame_height
    )

)


if not video_writer.isOpened():

    print(
        "ERROR: Could not create video file."
    )

    cap.release()

    raise SystemExit


print(
    f"Recording video to: "
    f"{RECORDED_VIDEO}"
)

print(
    "Press Q to stop the session."
)


# ============================================================
# SESSION VARIABLES
# ============================================================

session_data = []

frame_number = 0

face_detected_frames = 0

blink_count = 0

eyes_closed = False


gaze_counts = {

    "LEFT": 0,

    "CENTER": 0,

    "RIGHT": 0

}


gaze_ratios = []


start_time = cv2.getTickCount()


# ============================================================
# CREATE FACE LANDMARKER
# ============================================================

with FaceLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        success, frame = cap.read()


        if not success:

            print(
                "Could not read frame."
            )

            break


        frame_number += 1


        height, width = (
            frame.shape[:2]
        )


        # ----------------------------------------------------
        # Clean frame for OpenFace
        # ----------------------------------------------------

        clean_frame = frame.copy()

        video_writer.write(
            clean_frame
        )


        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2RGB

        )


        mp_image = mp.Image(

            image_format=(
                mp.ImageFormat.SRGB
            ),

            data=rgb_frame

        )


        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        timestamp_ms = int(

            (frame_number / VIDEO_FPS)
            * 1000

        )


        result = (
            landmarker.detect_for_video(

                mp_image,

                timestamp_ms

            )
        )


        # ====================================================
        # DEFAULT VALUES
        # ====================================================

        ear = np.nan

        pitch = np.nan

        yaw = np.nan

        roll = np.nan

        gaze_ratio = np.nan

        gaze_label = "UNKNOWN"

        face_detected = 0


        # ====================================================
        # FACE DETECTED
        # ====================================================

        if result.face_landmarks:

            face_detected = 1

            face_detected_frames += 1


            face_landmarks = (
                result.face_landmarks[0]
            )


            # ------------------------------------------------
            # Convert landmarks
            # ------------------------------------------------

            points = []


            for landmark in face_landmarks:

                x = (
                    landmark.x
                    * width
                )

                y = (
                    landmark.y
                    * height
                )


                points.append(
                    (x, y)
                )


                cv2.circle(

                    frame,

                    (
                        int(x),
                        int(y)
                    ),

                    1,

                    (0, 255, 0),

                    -1

                )


            # =================================================
            # EAR
            # =================================================

            left_ear = calculate_ear(

                points,

                LEFT_EYE

            )


            right_ear = calculate_ear(

                points,

                RIGHT_EYE

            )


            ear = (

                left_ear
                + right_ear

            ) / 2.0


            # =================================================
            # BLINK
            # =================================================

            if ear < EAR_THRESHOLD:

                if not eyes_closed:

                    eyes_closed = True

            else:

                if eyes_closed:

                    blink_count += 1

                    eyes_closed = False


            # =================================================
            # HEAD POSE
            # =================================================

            pose = calculate_head_pose(

                points,

                width,

                height

            )


            if pose is not None:

                (
                    raw_pitch,
                    raw_yaw,
                    raw_roll
                ) = pose


                pitch = smooth_value(

                    pitch_history,

                    raw_pitch

                )


                yaw = smooth_value(

                    yaw_history,

                    raw_yaw

                )


                roll = smooth_value(

                    roll_history,

                    raw_roll

                )


            # =================================================
            # GAZE
            # =================================================

            (
                raw_gaze_label,
                raw_gaze_ratio
            ) = calculate_gaze(points)


            if not np.isnan(
                raw_gaze_ratio
            ):

                gaze_ratio = (
                    smooth_value(

                        gaze_history,

                        raw_gaze_ratio

                    )
                )


                if (
                    gaze_ratio
                    < GAZE_LEFT_THRESHOLD
                ):

                    gaze_label = "LEFT"

                elif (
                    gaze_ratio
                    > GAZE_RIGHT_THRESHOLD
                ):

                    gaze_label = "RIGHT"

                else:

                    gaze_label = "CENTER"


                gaze_counts[
                    gaze_label
                ] += 1


                gaze_ratios.append(
                    gaze_ratio
                )


            # =================================================
            # DISPLAY
            # =================================================

            cv2.putText(

                frame,

                "FACE DETECTED",

                (20, 45),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 0),

                2

            )


            cv2.putText(

                frame,

                f"EAR: {ear:.3f}",

                (20, 90),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (255, 255, 255),

                2

            )


            cv2.putText(

                frame,

                f"Blinks: {blink_count}",

                (20, 125),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (255, 255, 255),

                2

            )


            if not np.isnan(pitch):

                cv2.putText(

                    frame,

                    f"Pitch: {pitch:.2f} deg",

                    (20, 175),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    (255, 255, 255),

                    2

                )


                cv2.putText(

                    frame,

                    f"Yaw: {yaw:.2f} deg",

                    (20, 210),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    (255, 255, 255),

                    2

                )


                cv2.putText(

                    frame,

                    f"Roll: {roll:.2f} deg",

                    (20, 245),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    (255, 255, 255),

                    2

                )


            cv2.putText(

                frame,

                f"Gaze: {gaze_label}",

                (20, 290),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                (255, 255, 255),

                2

            )


            if not np.isnan(
                gaze_ratio
            ):

                cv2.putText(

                    frame,

                    f"Gaze Ratio: "
                    f"{gaze_ratio:.3f}",

                    (20, 325),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    (255, 255, 255),

                    2

                )


        else:

            cv2.putText(

                frame,

                "NO FACE",

                (20, 45),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 0, 255),

                2

            )


        # ====================================================
        # SAVE FRAME DATA
        # ====================================================

        session_data.append([

            frame_number,

            face_detected,

            ear,

            pitch,

            yaw,

            roll,

            gaze_label,

            gaze_ratio,

            blink_count

        ])


        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(

            "ABDA - Facial Pipeline",

            frame

        )


        key = (
            cv2.waitKey(1)
            & 0xFF
        )


        if key == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

video_writer.release()

cv2.destroyAllWindows()


# ============================================================
# SESSION CALCULATIONS
# ============================================================

end_time = cv2.getTickCount()


duration_seconds = (

    end_time - start_time

) / cv2.getTickFrequency()


if duration_seconds <= 0:

    duration_seconds = 1


total_frames = len(
    session_data
)


face_quality = (

    face_detected_frames
    / total_frames
    * 100

    if total_frames > 0

    else 0

)


# ============================================================
# VALID VALUES
# ============================================================

def valid_values(
    column_index
):

    values = []


    for row in session_data:

        value = row[column_index]


        if value is not None:

            try:

                numeric_value = float(
                    value
                )


                if np.isfinite(
                    numeric_value
                ):

                    values.append(
                        numeric_value
                    )

            except (
                TypeError,
                ValueError
            ):

                pass


    return np.array(
        values,
        dtype=float
    )


ear_values = valid_values(2)

pitch_values = valid_values(3)

yaw_values = valid_values(4)

roll_values = valid_values(5)

gaze_ratio_values = valid_values(7)


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(
    values
):

    if len(values) == 0:

        return {

            "mean": np.nan,

            "std": np.nan,

            "min": np.nan,

            "max": np.nan

        }


    return {

        "mean": float(
            np.mean(values)
        ),

        "std": float(
            np.std(values)
        ),

        "min": float(
            np.min(values)
        ),

        "max": float(
            np.max(values)
        )

    }


ear_stats = calculate_statistics(
    ear_values
)

pitch_stats = calculate_statistics(
    pitch_values
)

yaw_stats = calculate_statistics(
    yaw_values
)

roll_stats = calculate_statistics(
    roll_values
)

gaze_stats = calculate_statistics(
    gaze_ratio_values
)


# ============================================================
# GAZE PERCENTAGES
# ============================================================

total_gaze_frames = sum(
    gaze_counts.values()
)


if total_gaze_frames > 0:

    left_percent = (

        gaze_counts["LEFT"]
        / total_gaze_frames

    ) * 100


    center_percent = (

        gaze_counts["CENTER"]
        / total_gaze_frames

    ) * 100


    right_percent = (

        gaze_counts["RIGHT"]
        / total_gaze_frames

    ) * 100

else:

    left_percent = 0

    center_percent = 0

    right_percent = 0


# ============================================================
# BLINK RATE
# ============================================================

blink_rate = (

    blink_count
    /
    (duration_seconds / 60)

)


# ============================================================
# SAVE FACIAL SESSION CSV
# ============================================================

with open(

    CSV_FILE,

    "w",

    newline="",

    encoding="utf-8"

) as file:

    writer = csv.writer(file)


    writer.writerow([

        "Frame",

        "FaceDetected",

        "EAR",

        "Pitch",

        "Yaw",

        "Roll",

        "Gaze",

        "GazeRatio",

        "CumulativeBlinks"

    ])


    writer.writerows(
        session_data
    )


# ============================================================
# RUN OPENFACE
# ============================================================

openface_csv = run_openface(
    RECORDED_VIDEO
)


# ============================================================
# ANALYZE AUs
# ============================================================

au_statistics = (
    analyze_action_units(
        openface_csv
    )
)


# ============================================================
# SAVE AU SUMMARY CSV
# ============================================================

save_au_summary(
    au_statistics
)


# ============================================================
# SAVE MAIN SUMMARY
# ============================================================

with open(

    SUMMARY_FILE,

    "w",

    encoding="utf-8"

) as file:

    file.write(
        "========================================\n"
    )

    file.write(
        "       ABDA FACIAL SESSION SUMMARY\n"
    )

    file.write(
        "========================================\n\n"
    )


    # --------------------------------------------------------
    # SESSION
    # --------------------------------------------------------

    file.write(
        f"Duration          : "
        f"{duration_seconds:.2f} sec\n"
    )

    file.write(
        f"Total Frames      : "
        f"{total_frames}\n"
    )

    file.write(
        f"Face Detected     : "
        f"{face_detected_frames}\n"
    )

    file.write(
        f"Face Quality      : "
        f"{face_quality:.2f} %\n\n"
    )


    # --------------------------------------------------------
    # HEAD POSE
    # --------------------------------------------------------

    file.write(
        "HEAD POSE\n"
    )

    file.write(
        "----------------------------------------\n"
    )

    file.write(
        f"Pitch Mean        : "
        f"{pitch_stats['mean']:.2f} deg\n"
    )

    file.write(
        f"Pitch Std         : "
        f"{pitch_stats['std']:.2f} deg\n"
    )

    file.write(
        f"Pitch Min         : "
        f"{pitch_stats['min']:.2f} deg\n"
    )

    file.write(
        f"Pitch Max         : "
        f"{pitch_stats['max']:.2f} deg\n\n"
    )


    file.write(
        f"Yaw Mean          : "
        f"{yaw_stats['mean']:.2f} deg\n"
    )

    file.write(
        f"Yaw Std           : "
        f"{yaw_stats['std']:.2f} deg\n"
    )

    file.write(
        f"Yaw Min           : "
        f"{yaw_stats['min']:.2f} deg\n"
    )

    file.write(
        f"Yaw Max           : "
        f"{yaw_stats['max']:.2f} deg\n\n"
    )


    file.write(
        f"Roll Mean         : "
        f"{roll_stats['mean']:.2f} deg\n"
    )

    file.write(
        f"Roll Std          : "
        f"{roll_stats['std']:.2f} deg\n"
    )

    file.write(
        f"Roll Min          : "
        f"{roll_stats['min']:.2f} deg\n"
    )

    file.write(
        f"Roll Max          : "
        f"{roll_stats['max']:.2f} deg\n\n"
    )


    # --------------------------------------------------------
    # BLINK
    # --------------------------------------------------------

    file.write(
        "BLINK\n"
    )

    file.write(
        "----------------------------------------\n"
    )

    file.write(
        f"Total Blinks      : "
        f"{blink_count}\n"
    )

    file.write(
        f"Blink Rate        : "
        f"{blink_rate:.2f} / min\n\n"
    )


    # --------------------------------------------------------
    # EAR
    # --------------------------------------------------------

    file.write(
        "EAR\n"
    )

    file.write(
        "----------------------------------------\n"
    )

    file.write(
        f"EAR Mean          : "
        f"{ear_stats['mean']:.3f}\n"
    )

    file.write(
        f"EAR Std           : "
        f"{ear_stats['std']:.3f}\n"
    )

    file.write(
        f"EAR Min           : "
        f"{ear_stats['min']:.3f}\n"
    )

    file.write(
        f"EAR Max           : "
        f"{ear_stats['max']:.3f}\n\n"
    )


    # --------------------------------------------------------
    # GAZE
    # --------------------------------------------------------

    file.write(
        "GAZE\n"
    )

    file.write(
        "----------------------------------------\n"
    )

    file.write(
        f"Left              : "
        f"{left_percent:.2f} %\n"
    )

    file.write(
        f"Center            : "
        f"{center_percent:.2f} %\n"
    )

    file.write(
        f"Right             : "
        f"{right_percent:.2f} %\n"
    )

    file.write(
        f"Gaze Ratio Mean   : "
        f"{gaze_stats['mean']:.3f}\n"
    )

    file.write(
        f"Gaze Ratio Std    : "
        f"{gaze_stats['std']:.3f}\n"
    )

    file.write(
        f"Gaze Ratio Min    : "
        f"{gaze_stats['min']:.3f}\n"
    )

    file.write(
        f"Gaze Ratio Max    : "
        f"{gaze_stats['max']:.3f}\n\n"
    )


    # --------------------------------------------------------
    # ACTION UNITS
    # --------------------------------------------------------

    file.write(
        "ACTION UNITS\n"
    )

    file.write(
        "----------------------------------------\n"
    )


    if au_statistics:

        intensity_aus = [

            name

            for name in au_statistics

            if name.endswith("_r")

        ]


        if intensity_aus:

            file.write(
                "\nAU INTENSITY\n"
            )


            for au_name in sorted(
                intensity_aus
            ):

                stats = (
                    au_statistics[au_name]
                )


                file.write(

                    f"{au_name:<10} "

                    f"Mean: "
                    f"{stats['mean']:.3f}   "

                    f"Std: "
                    f"{stats['std']:.3f}   "

                    f"Min: "
                    f"{stats['min']:.3f}   "

                    f"Max: "
                    f"{stats['max']:.3f}   "

                    f"Activity: "
                    f"{stats['activity_rate']:.2f}%\n"

                )


        presence_aus = [

            name

            for name in au_statistics

            if name.endswith("_c")

        ]


        if presence_aus:

            file.write(
                "\nAU PRESENCE\n"
            )


            for au_name in sorted(
                presence_aus
            ):

                stats = (
                    au_statistics[au_name]
                )


                file.write(

                    f"{au_name:<10} "

                    f"Presence: "
                    f"{stats['presence_rate']:.2f}%\n"

                )

    else:

        file.write(
            "OpenFace AU analysis "
            "not available.\n"
        )


    # --------------------------------------------------------
    # OpenFace file location
    # --------------------------------------------------------

    if openface_csv:

        file.write(
            "\nOpenFace Raw CSV:\n"
        )

        file.write(
            f"{openface_csv}\n"
        )


    file.write(
        "\n========================================\n"
    )


# ============================================================
# PRINT MAIN SUMMARY
# ============================================================

print("\n")

print(
    "========================================"
)

print(
    "       ABDA FACIAL SESSION SUMMARY"
)

print(
    "========================================"
)


print(
    f"\nDuration          : "
    f"{duration_seconds:.2f} sec"
)

print(
    f"Total Frames      : "
    f"{total_frames}"
)

print(
    f"Face Detected     : "
    f"{face_detected_frames}"
)

print(
    f"Face Quality      : "
    f"{face_quality:.2f} %"
)


# ============================================================
# HEAD POSE
# ============================================================

print(
    "\nHEAD POSE"
)

print(
    "----------------------------------------"
)

print(
    f"Pitch Mean        : "
    f"{pitch_stats['mean']:.2f} deg"
)

print(
    f"Pitch Std         : "
    f"{pitch_stats['std']:.2f} deg"
)

print(
    f"Pitch Min         : "
    f"{pitch_stats['min']:.2f} deg"
)

print(
    f"Pitch Max         : "
    f"{pitch_stats['max']:.2f} deg"
)

print()

print(
    f"Yaw Mean          : "
    f"{yaw_stats['mean']:.2f} deg"
)

print(
    f"Yaw Std           : "
    f"{yaw_stats['std']:.2f} deg"
)

print(
    f"Yaw Min           : "
    f"{yaw_stats['min']:.2f} deg"
)

print(
    f"Yaw Max           : "
    f"{yaw_stats['max']:.2f} deg"
)

print()

print(
    f"Roll Mean         : "
    f"{roll_stats['mean']:.2f} deg"
)

print(
    f"Roll Std          : "
    f"{roll_stats['std']:.2f} deg"
)

print(
    f"Roll Min          : "
    f"{roll_stats['min']:.2f} deg"
)

print(
    f"Roll Max          : "
    f"{roll_stats['max']:.2f} deg"
)


# ============================================================
# BLINK
# ============================================================

print(
    "\nBLINK"
)

print(
    "----------------------------------------"
)

print(
    f"Total Blinks      : "
    f"{blink_count}"
)

print(
    f"Blink Rate        : "
    f"{blink_rate:.2f} / min"
)


# ============================================================
# EAR
# ============================================================

print(
    "\nEAR"
)

print(
    "----------------------------------------"
)

print(
    f"EAR Mean          : "
    f"{ear_stats['mean']:.3f}"
)

print(
    f"EAR Std           : "
    f"{ear_stats['std']:.3f}"
)

print(
    f"EAR Min           : "
    f"{ear_stats['min']:.3f}"
)

print(
    f"EAR Max           : "
    f"{ear_stats['max']:.3f}"
)


# ============================================================
# GAZE
# ============================================================

print(
    "\nGAZE"
)

print(
    "----------------------------------------"
)

print(
    f"Left              : "
    f"{left_percent:.2f} %"
)

print(
    f"Center            : "
    f"{center_percent:.2f} %"
)

print(
    f"Right             : "
    f"{right_percent:.2f} %"
)

print(
    f"Gaze Ratio Mean   : "
    f"{gaze_stats['mean']:.3f}"
)

print(
    f"Gaze Ratio Std    : "
    f"{gaze_stats['std']:.3f}"
)

print(
    f"Gaze Ratio Min    : "
    f"{gaze_stats['min']:.3f}"
)

print(
    f"Gaze Ratio Max    : "
    f"{gaze_stats['max']:.3f}"
)


# ============================================================
# AU SUMMARY
# ============================================================

print_au_summary(
    au_statistics
)


# ============================================================
# FINAL FILES
# ============================================================

print(
    "\n========================================"
)

print(
    "\nSaved:"
)

print(
    f"  {CSV_FILE}"
)

print(
    f"  {SUMMARY_FILE}"
)

print(
    f"  {RECORDED_VIDEO}"
)

if au_statistics:

    print(
        f"  {AU_SUMMARY_FILE}"
    )


if openface_csv:

    print(
        f"  OpenFace CSV: "
        f"{openface_csv}"
    )


print(
    "\nSession completed."
)