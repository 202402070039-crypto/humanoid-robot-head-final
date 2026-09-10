import soundfile as sf
import sounddevice as sd
import numpy as np
import time
import hardware_control


# =====================================================
# AUDIO
# =====================================================

AUDIO_FILE = "robot_response.wav"

CHUNK_MS = 40


# =====================================================
# JAW COMMANDS
# =====================================================

JAW_CLOSED = 45
JAW_MIDDLE = 90
JAW_OPEN = 120


# =====================================================
# AUDIO THRESHOLDS
# =====================================================

SPEECH_THRESHOLD = 0.025
HIGH_THRESHOLD = 0.15


# =====================================================
# SMOOTHING
# =====================================================

SMOOTHING_FACTOR = 0.35

previous_level = 0.0


# =====================================================
# MIDDLE-JAW ALTERNATION
# =====================================================

# Time between 45 <-> 90 changes
# during medium-amplitude speech.

JAW_ALTERNATE_INTERVAL = 0.12


# =====================================================
# PLAY SPEECH + JAW
# =====================================================

def play_speech_with_jaw():

    global previous_level

    print("Starting robot speech + jaw synchronization...")

    audio_file = sf.SoundFile(
        AUDIO_FILE,
        mode="r"
    )

    sample_rate = audio_file.samplerate
    channels = audio_file.channels

    chunk_frames = int(
        sample_rate * CHUNK_MS / 1000
    )

    stream = sd.OutputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="float32"
    )

    stream.start()

    previous_level = 0.0

    # -------------------------------------------------
    # Jaw state
    # -------------------------------------------------

    previous_jaw_position = JAW_CLOSED

    alternate_jaw_position = JAW_CLOSED

    last_alternate_time = time.monotonic()


    try:

        while True:

            # =========================================
            # Read audio chunk
            # =========================================

            data = audio_file.read(
                chunk_frames,
                dtype="float32"
            )

            if len(data) == 0:
                break


            # =========================================
            # Calculate RMS
            # =========================================

            rms = np.sqrt(
                np.mean(data ** 2)
            )

            level = float(rms)


            # =========================================
            # Smooth audio level
            # =========================================

            smoothed_level = (
                SMOOTHING_FACTOR * level
                +
                (1 - SMOOTHING_FACTOR) * previous_level
            )

            previous_level = smoothed_level


            # =========================================
            # Determine jaw behavior
            # =========================================

            current_time = time.monotonic()


            # -----------------------------------------
            # LOW AMPLITUDE
            # -----------------------------------------

            if smoothed_level < SPEECH_THRESHOLD:

                jaw_position = JAW_CLOSED

                alternate_jaw_position = JAW_CLOSED


            # -----------------------------------------
            # HIGH AMPLITUDE
            # -----------------------------------------

            elif smoothed_level >= HIGH_THRESHOLD:

                jaw_position = JAW_OPEN

                alternate_jaw_position = JAW_CLOSED

                last_alternate_time = current_time


            # -----------------------------------------
            # MEDIUM AMPLITUDE
            # -----------------------------------------

            else:

                # Alternate between CLOSED and MIDDLE

                if (
                    current_time - last_alternate_time
                    >= JAW_ALTERNATE_INTERVAL
                ):

                    if (
                        alternate_jaw_position
                        == JAW_CLOSED
                    ):
                        alternate_jaw_position = JAW_MIDDLE

                    else:
                        alternate_jaw_position = JAW_CLOSED

                    last_alternate_time = current_time

                jaw_position = alternate_jaw_position


            # =========================================
            # SEND ONLY WHEN POSITION CHANGES
            # =========================================

            if jaw_position != previous_jaw_position:

                hardware_control.send_jaw(
                    jaw_position
                )

                previous_jaw_position = jaw_position


            # =========================================
            # PLAY AUDIO
            # =========================================

            stream.write(
                data
            )


    finally:

        # =============================================
        # Stop audio
        # =============================================

        stream.stop()
        stream.close()

        audio_file.close()


        # =============================================
        # Close jaw
        # =============================================

        hardware_control.send_jaw(
            JAW_CLOSED
        )

        print("Robot speech finished.")