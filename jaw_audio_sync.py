import soundfile as sf
import sounddevice as sd
import numpy as np

import robot_control


# =========================================================
# AUDIO SETTINGS
# =========================================================

AUDIO_FILE = "robot_response.wav"

CHUNK_MS = 40


# =========================================================
# JAW SETTINGS
# =========================================================

JAW_CLOSED = 60
JAW_MIDDLE = 90
JAW_OPEN = 120


# =========================================================
# AUDIO THRESHOLDS
# =========================================================

SPEECH_THRESHOLD = 0.025

HIGH_THRESHOLD = 0.15


# =========================================================
# SMOOTHING
# =========================================================

SMOOTHING_FACTOR = 0.35

previous_level = 0.0


# =========================================================
# PLAY AUDIO + CONTROL JAW
# =========================================================

def play_speech_with_jaw():

    global previous_level

    print("Starting robot speech + jaw synchronization...")

    # Open WAV file
    audio_file = sf.SoundFile(
        AUDIO_FILE,
        mode="r"
    )

    sample_rate = audio_file.samplerate

    channels = audio_file.channels

    chunk_frames = int(
        sample_rate * CHUNK_MS / 1000
    )


    # Create audio output stream
    stream = sd.OutputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="float32"
    )

    stream.start()


    # Reset smoothing
    previous_level = 0.0


    # =====================================================
    # AUDIO LOOP
    # =====================================================

    while True:

        data = audio_file.read(
            chunk_frames,
            dtype="float32"
        )

        if len(data) == 0:

            break


        # -------------------------------------------------
        # Calculate RMS amplitude
        # -------------------------------------------------

        rms = np.sqrt(
            np.mean(
                data ** 2
            )
        )


        level = float(rms)


        # -------------------------------------------------
        # Smooth audio level
        # -------------------------------------------------

        smoothed_level = (
            SMOOTHING_FACTOR * level
            +
            (1 - SMOOTHING_FACTOR) * previous_level
        )

        previous_level = smoothed_level


        # -------------------------------------------------
        # Three-position jaw decision
        # -------------------------------------------------

        if smoothed_level < SPEECH_THRESHOLD:

            jaw_position = JAW_CLOSED

        elif smoothed_level < HIGH_THRESHOLD:

            jaw_position = JAW_MIDDLE

        else:

            jaw_position = JAW_OPEN


        # -------------------------------------------------
        # Send jaw position through MQTT
        # -------------------------------------------------

        robot_control.send_jaw(
            jaw_position
        )


        # -------------------------------------------------
        # Play audio
        # -------------------------------------------------

        stream.write(data)


    # =====================================================
    # FINISH
    # =====================================================

    stream.stop()

    stream.close()

    audio_file.close()


    # Close jaw
    robot_control.send_jaw(
        JAW_CLOSED
    )

    print("Robot speech finished.")