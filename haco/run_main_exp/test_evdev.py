import evdev
import time

def find_g29():
    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    for device in devices:
        if "Logitech G29 Driving Force Racing Wheel" in device.name:
            print(f"Found G29 at {device.path}")
            return device
    return None

def create_force_feedback(device):
    """ Uploads a constant force feedback effect to the G29. """
    effect = evdev.ff.Effect(
        type=evdev.ecodes.FF_CONSTANT,  # Force Feedback Type
        id=-1,  # New effect ID
        trigger=evdev.ff.Trigger(0, 0),  # No trigger buttons
        replay=evdev.ff.Replay(1000, 0),  # Duration: 1s, no delay
        direction=0,  # 0 degrees (forward)
        constant=evdev.ff.Constant(32767*1000, evdev.ff.Envelope(0, 0, 0, 0))  # Full strength force
    )

    effect_id = device.upload_effect(effect)
    print(f"Effect uploaded with ID: {effect_id}")
    return effect_id
def play_effect(device, effect_id):
    """ Plays the uploaded force feedback effect. """
    device.write(evdev.ecodes.EV_FF, effect_id, 1)
    print("Force feedback activated")

if __name__ == "__main__":
    device = find_g29()
    if not device:
        print("G29 not found. Ensure it's connected and accessible.")
    else:
        effect_id = create_force_feedback(device)
        play_effect(device, effect_id)
        time.sleep(200)  # Keep force for 2 seconds
        device.erase_effect(effect_id)  # Cleanup the effect

