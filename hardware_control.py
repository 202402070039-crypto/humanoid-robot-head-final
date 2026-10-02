import paho.mqtt.client as mqtt
import time

# =====================================================
# PRIVATE HIVEMQ CLOUD
# =====================================================

BROKER = "3f687bb3e1da49adb7e4e7b5a183432b.s1.eu.hivemq.cloud"
PORT = 8883

USERNAME = "Robot_Head_Python"
PASSWORD = "vstechhorizon"

# =====================================================
# MQTT TOPICS
# =====================================================

JAW_TOPIC = "humanoid_robot_piyusha/jaw"

EYE_HORIZONTAL_TOPIC = "humanoid_robot_piyusha/eye_horizontal"

EYE_VERTICAL_TOPIC = "humanoid_robot_piyusha/eye_vertical"

NECK_TOPIC = "humanoid_robot_piyusha/neck"

# =====================================================
# SERVO COMMAND LIMITS
# =====================================================

JAW_MIN = 45
JAW_MAX = 120

EYE_HORIZONTAL_MIN = 45
EYE_HORIZONTAL_MAX = 135

EYE_VERTICAL_MIN = 90
EYE_VERTICAL_MAX = 180

NECK_MIN = 45
NECK_MAX = 180

# =====================================================
# MQTT CLIENT
# =====================================================

client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2
)

client.username_pw_set(
    USERNAME,
    PASSWORD
)

client.tls_set()

# =====================================================
# CONNECT
# =====================================================

def connect():

    if client.is_connected():
        return

    print(
        "Connecting to private HiveMQ Cloud..."
    )

    try:

        client.connect(
            BROKER,
            PORT,
            60
        )

        client.loop_start()

        time.sleep(0.5)

        if client.is_connected():

            print(
                "Private MQTT connected."
            )

        else:

            print(
                "Private MQTT connection could "
                "not be confirmed."
            )

    except Exception as e:

        print(
            "MQTT connection error:"
        )

        print(e)


# =====================================================
# INTERNAL PUBLISH FUNCTION
# =====================================================

def _publish(topic, value):

    if not client.is_connected():

        print(
            "MQTT not connected. "
            "Servo command skipped."
        )

        return False

    result = client.publish(
        topic,
        str(value),
        qos=0
    )

    if result.rc != mqtt.MQTT_ERR_SUCCESS:

        print(
            f"MQTT publish failed: "
            f"{result.rc}"
        )

        return False

    return True


# =====================================================
# SEND JAW
# =====================================================

def send_jaw(position):

    if position < JAW_MIN or position > JAW_MAX:

        print(
            "Invalid jaw position:",
            position
        )

        return

    _publish(
        JAW_TOPIC,
        position
    )


# =====================================================
# SEND HORIZONTAL EYE
# =====================================================
#
# 45  = robot RIGHT
# 90  = CENTER
# 135 = robot LEFT
#

def send_eye_horizontal(angle):

    if (
        angle < EYE_HORIZONTAL_MIN
        or
        angle > EYE_HORIZONTAL_MAX
    ):

        print(
            "Invalid horizontal eye angle:",
            angle
        )

        return

    _publish(
        EYE_HORIZONTAL_TOPIC,
        angle
    )


# =====================================================
# SEND VERTICAL EYE
# =====================================================
#
# 90  = UP
# 135 = CENTER
# 180 = DOWN
#

def send_eye_vertical(angle):

    if (
        angle < EYE_VERTICAL_MIN
        or
        angle > EYE_VERTICAL_MAX
    ):

        print(
            "Invalid vertical eye angle:",
            angle
        )

        return

    _publish(
        EYE_VERTICAL_TOPIC,
        angle
    )


# =====================================================
# SEND NECK
# =====================================================
#
# Current calibrated usable range:
# 90 to 180
#

def send_neck(angle):

    if (
        angle < NECK_MIN
        or
        angle > NECK_MAX
    ):

        print(
            "Invalid neck angle:",
            angle
        )

        return

    _publish(
        NECK_TOPIC,
        angle
    )


# =====================================================
# DISCONNECT
# =====================================================

def disconnect():

    if not client.is_connected():
        return

    client.loop_stop()

    client.disconnect()

    print(
        "Private MQTT disconnected."
    )
