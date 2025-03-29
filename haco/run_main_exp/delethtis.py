import evdev
from evdev import ecodes
from fcntl import ioctl
import struct
import sys

# Find the Logitech G29 with force feedback support.
devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
g29 = next((d for d in devices if "Logitech G29" in d.name and ecodes.EV_FF in d.capabilities()), None)

if not g29:
    print("G29 with FF not found.")
    sys.exit(1)

print(f"Using device: {g29.path}")

EVIOCSFF = 0x40304580
FF_CONSTANT = 0x50    
effect_format = "<HhHHHHHhHHHH"  # Ensure struct format matches the kernel's ff_effect

# Effect parameters with stronger force and infinite duration
effect_bytes = struct.pack(
    effect_format,
    FF_CONSTANT,     # type
    -1,              # id (kernel assigns a free ID if -1)
    0,               # direction (0 degrees)
    0,               # trigger.button (none)
    0,               # trigger.interval 
    0,               # replay.length (0 = infinite duration)
    0,               # replay.delay (0 ms)
    0x6000,          # constant.level (stronger force, signed 16-bit)
    0,               # envelope.attack_length
    0,               # envelope.attack_level
    0,               # envelope.fade_length
    0                # envelope.fade_level
)

effect_buf = bytearray(effect_bytes)

try:
    # Upload effect to the device
    effect_id = ioctl(g29.fd, EVIOCSFF, effect_buf)
    print(f"Created effect with ID: {effect_id}")

    # Set FF gain to maximum (critical for effect strength)
    g29.write(ecodes.EV_FF, ecodes.FF_GAIN, 0xFFFF)
    g29.syn()

    # Play the effect
    g29.write(ecodes.EV_FF, effect_id, 1)
    g29.syn()
except OSError as e:
    print(f"Failed to upload/play effect: {e}")
    sys.exit(1)

print("Force feedback should now be active. Press CTRL+C to stop.")

try:
    for event in g29.read_loop():
        # Optional: Handle events or monitor device feedback
        pass
except KeyboardInterrupt:
    print("\nStopping effect...")
    # Stop the effect and reset gain
    g29.write(ecodes.EV_FF, effect_id, 0)
    g29.write(ecodes.EV_FF, ecodes.FF_GAIN, 0)
    g29.syn()