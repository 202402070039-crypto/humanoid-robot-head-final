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

# =====================================================
# MQTT CLIENT
# =====================================================

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

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

    print("Connecting to private HiveMQ Cloud...")

    client.connect(
        BROKER,
        PORT,
        60
    )

    client.loop_start()

    time.sleep(0.5)

    print("Private MQTT connected.")


# =====================================================
# SEND JAW
# =====================================================

def send_jaw(position):

    if position not in [45, 90, 120]:
        print("Invalid jaw position:", position)
        return

    if not client.is_connected():
        print("MQTT not connected. Jaw command skipped.")
        return

    client.publish(
        JAW_TOPIC,
        str(position)
    )


# =====================================================
# DISCONNECT
# =====================================================

def disconnect():

    if not client.is_connected():
        return

    client.loop_stop()
    client.disconnect()

    print("Private MQTT disconnected.")