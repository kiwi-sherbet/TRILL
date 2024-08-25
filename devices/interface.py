import time
import threading
import numpy as np
import hid
from pynput import mouse, keyboard


## Define the thread receiving keyboard for debugging
class Keyboard:

    def __init__(self, pos_sensitivity=6.0, rot_sensitivity=6.0,
                 right_gripper=True, left_gripper=False):

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self._left_gripper = right_gripper
        self._right_gripper = left_gripper

        # 3-DOF variables
        self._x, self._y, self._yaw = 0, 0, 0

        self.single_click_and_hold = False

        self._control = np.zeros(6)
        self._enabled = False

        self._rebase = False
        self._grasp_r = False
        self._grasp_l = False
        self._reset = False

        self._flag_init = False
        self._t_last_click = -1
        self._t_click = -1

        # launch a new listener thread to listen to keyboard
        self.thread = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self.thread.daemon = True

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        # Reset 6-DOF variables
        self._x, self._y, self._yaw = 0, 0, 0
        # Reset grasp
        self.single_click_and_hold = False

        self._flag_init = False
        self._t_last_click = -1
        self._t_click = -1

    def _on_press(self, key):

        try:
            key_char = key.char
        except AttributeError:
            key_char = str(key)

        if key_char == "e":
            self._t_last_click = -1
            self._t_click = time.time()
            elapsed_time = self._t_click - self._t_last_click
            self._t_last_click = self._t_click
            self.single_click_and_hold = True

        elif key_char == "w":
            self._x = min(1, self._x + 1)
        elif key_char == "s":
            self._x = max(-1, self._x - 1)
        elif key_char == "a":
            self._y = min(1, self._y + 1)
        elif key_char == "d":
            self._y = max(-1, self._y - 1)
        elif key_char == "x":
            self._yaw = max(-1, self._yaw - 1)
        elif key_char == "z":
            self._yaw = max(1, self._yaw + 1)
        elif key_char == "g":
            self._grasp_r = True
        elif key_char == "f":
            self._grasp_l = True
        elif key_char == "r":
            self._rebase = not self._rebase
        elif key_char == "t":
            self._reset = True
        elif key_char == "q":
            self._enabled = False
        elif key_char == "q" or key == keyboard.Key.esc:
            self._enabled = False

        print("Key pressed: {}".format(key))

    def _on_release(self, key):

        try:
            key_char = key.char
        except AttributeError:
            key_char = str(key)

        if key_char == "r":
            self._enabled = True
        if key_char == "g":
            self._grasp_r = False
        if key_char == "f":
            self._grasp_l = False
        if key_char == "e":
            self.single_click_and_hold = False

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        # Reset 3-DOF variables
        self._x, self._y, self._yaw = 0, 0, 0
        # Reset control
        self._control = np.zeros(6)
        # Reset grasp
        self.single_click_and_hold = False

    def start(self):
        self.thread.start()

    @property
    def control(self):
        """
        Grabs current pose of Spacemouse
        Returns:
            np.array: 6-DoF control value
        """
        return_val = (self._x, self._y, self._yaw)
        self._x, self._y, self._yaw = 0, 0, 0
        return return_val

    @property
    def click(self):
        """
        Maps internal states into gripper commands.
        Returns:
            float: Whether we're using single click and hold or not
        """
        if self.single_click_and_hold:
            return 1.0
        return 0

    @property
    def enable(self):
        return self._enabled

    @property
    def grasp(self):
        # only available for both hands or one hand
        if self._right_gripper and self._left_gripper:
            return self._grasp_r, self._grasp_l
        else:
            assert self._right_gripper or self._left_gripper
            if self._right_gripper:
                return self._grasp_r
            else:
                return self._grasp_l

    @property
    def rebase(self):
        return self._rebase

    @property
    def reset(self):
        if self._reset:
            self._reset = False
            return True
        else:
            return False


## Define the thread receiving keyboard for debugging ##
class Mouse:

    def __init__(self, pos_sensitivity=6.0, rot_sensitivity=6.0):

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        # 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self.single_click_and_hold = False

        self._control = np.zeros(4)
        self._reset_state = 0
        self._enabled = False

        self._flag_init = False
        self._t_last_click = -1
        self._t_click = -1

        # launch a new listener thread to listen to SpaceMouse
        self.thread = mouse.Listener(
            on_move=self._on_move, on_click=self._on_click, on_scroll=self._on_scroll
        )
        self.thread.start()

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        # Reset 6-DOF variables
        self.x, self.y = 0, 0
        self.roll, self.pitch = 0, 0
        # Reset control
        self._control = np.zeros(4)
        # Reset grasp
        self.single_click_and_hold = False

        self._flag_init = False
        self._t_last_click = -1
        self._t_click = -1

    def _on_move(self, x, y):
        if self._flag_init:
            self.x = x - self._x_offset
            self.y = y - self._y_offset
        else:
            self._x_offset = x
            self._y_offset = y
            self._flag_init = True

        self._control[0:2] = np.array([self.x, self.y]) / 100.0

    def _on_click(self, x, y, button, pressed):

        self._t_last_click = -1

        self._t_click = time.time()
        elapsed_time = self._t_click - self._t_last_click
        self._t_last_click = self._t_click
        self.single_click_and_hold = True

        # release left button
        if pressed == 0:
            self.single_click_and_hold = False

        # right button is for reset
        if pressed == 1:
            self._reset_state = 1
            self._enabled = False
            self._reset_internal_state()

    def _on_scroll(self, x, y, dx, dy):

        self.roll += dx
        self.pitch += dy

        self.roll = np.clip(self.roll, -1, 1)
        self.pitch = np.clip(self.pitch, -1, 1)

        self._control[2:4] = (
            self.pos_sensitivity * np.array([self.roll, -self.pitch]) / 10.0
        )

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        # Reset 6-DOF variables
        self.x, self.y = 0, 0
        self.roll, self.pitch = 0, 0
        # Reset control
        self._control = np.zeros(4)
        # Reset grasp
        self.single_click_and_hold = False

    @property
    def control(self):
        """
        Grabs current pose of Spacemouse
        Returns:
            np.array: 6-DoF control value
        """
        return self._control

    @property
    def click(self):
        """
        Maps internal states into gripper commands.
        Returns:
            float: Whether we're using single click and hold or not
        """
        if self.single_click_and_hold:
            return 1.0
        return 0


class SpaceMouse:
    """
    A minimalistic driver class for SpaceMouse with HID library.
    Note: Use hid.enumerate() to view all USB human interface devices (HID).
    Make sure SpaceMouse is detected before running the script.
    You can look up its vendor/product id from this method.
    Args:
        pos_sensitivity (float): Magnitude of input position command scaling
        rot_sensitivity (float): Magnitude of scale input rotation commands scaling
    """

    def __init__(self, pos_sensitivity=6.0, rot_sensitivity=6.0):

        print("Opening SpaceMouse device")
        found = False
        for device_info in hid.enumerate():
            if device_info["product_string"] == "SpaceMouse Wireless":
                self.info = device_info
                found = True
                break
        if found:
            print("Found SpaceMouse")
            print("Manufacturer: %s" % self.info["vendor_id"])
            print("Product: %s" % self.info["product_id"])
        else:
            print("Could not find SpaceMouse")
            return

        print("Opening SpaceMouse device")
        self.device = hid.Device(self.info["vendor_id"], self.info["product_id"])

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        # 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self.single_click_and_hold = False

        self._control = np.zeros(6)
        self._reset_state = 0
        self._enabled = False

        # launch a new listener thread to listen to SpaceMouse
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        # Reset 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0
        # Reset control
        self._control = np.zeros(6)
        # Reset grasp
        self.single_click_and_hold = False

    def run(self):
        """Listener method that keeps pulling new messages."""

        t_last_click = -1

        while True:
            d = self.device.read(13)

            if d is not None:

                if d[0] == 1:  ## readings from 6-DoF sensor
                    self.y = int.from_bytes(d[1:3], "little", signed=True)
                    self.x = int.from_bytes(d[3:5], "little", signed=True)
                    self.z = int.from_bytes(d[5:7], "little", signed=True)

                    self.roll = int.from_bytes(d[7:9], "little", signed=True)
                    self.pitch = int.from_bytes(d[9:11], "little", signed=True)
                    self.yaw = int.from_bytes(d[11:13], "little", signed=True)

                    self._control = (
                        -np.array(
                            [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]
                        )
                        / 350.0
                    )

                elif d[0] == 3:  ## readings from the side buttons

                    # press left button
                    if d[1] == 1:
                        t_click = time.time()
                        elapsed_time = t_click - t_last_click
                        t_last_click = t_click
                        self.single_click_and_hold = True

                    # release left button
                    if d[1] == 0:
                        self.single_click_and_hold = False

                    # right button is for reset
                    if d[1] == 2:
                        self._reset_state = 1
                        self._enabled = False
                        self._reset_internal_state()

    @property
    def control(self):
        """
        Grabs current pose of Spacemouse
        Returns:
            np.array: 6-DoF control value
        """
        return self._control

    @property
    def click(self):
        """
        Maps internal states into gripper commands.
        Returns:
            float: Whether we're using single click and hold or not
        """
        if self.single_click_and_hold:
            return 1.0
        return 0


if __name__ == "__main__":

    space_mouse = SpaceMouse()
    for i in range(100000):
        print(space_mouse.control)
        time.sleep(0.02)
