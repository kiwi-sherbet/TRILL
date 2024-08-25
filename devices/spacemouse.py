import threading
import time
import hid
import numpy as np

class SpaceMouse():
    """
    A minimalistic driver class for SpaceMouse with HID library.
    Note: Use hid.enumerate() to view all USB human interface devices (HID).
    Make sure SpaceMouse is detected before running the script.
    You can look up its vendor/product id from this method.
    Args:
        pos_sensitivity (float): Magnitude of input position command scaling
        rot_sensitivity (float): Magnitude of scale input rotation commands scaling
    """

    def __init__(self,
                 pos_sensitivity=0.004,
                 rot_sensitivity=1.0
                 ):

        print("Opening SpaceMouse device")
        found = False
        for device_info in hid.enumerate():
            print(device_info)
            if device_info['product_string'] == 'SpaceMouse Wireless':
                self.info = device_info
                found = True
                break
        if found:
            print("Found SpaceMouse")
            print("Manufacturer: %s" % self.info['vendor_id'])
            print("Product: %s" % self.info['product_id'])
        else:
            print("Could not find SpaceMouse")
            return

        print("Opening SpaceMouse device")
        self.device = hid.Device(self.info['vendor_id'], self.info['product_id'])

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
                    self.y += self.pos_sensitivity * int.from_bytes(d[1:3], "little", signed=True)
                    self.x += self.pos_sensitivity * int.from_bytes(d[3:5], "little", signed=True)
                    self.z += self.pos_sensitivity * int.from_bytes(d[5:7], "little", signed=True)

                    self.roll = self.rot_sensitivity * int.from_bytes(d[7:9], "little", signed=True)
                    self.pitch = self.rot_sensitivity * int.from_bytes(d[9:11], "little", signed=True)
                    self.yaw = self.rot_sensitivity * int.from_bytes(d[11:13], "little", signed=True)

                    self._control = - np.array([self.x,
                                                self.y,
                                                self.z,
                                                self.roll,
                                                self.pitch,
                                                self.yaw])/350.

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
