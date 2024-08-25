import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)

import numpy as np
from util import geom
from simulator.envs import DoorEnv, EmptyEnv, Kitchen2Env, KitchenEnv
from simulator.render import CV2Renderer
from simulator.recorder import HDF5Recorder
import time
import argparse

from devices.spacemouse import SpaceMouse
from devices.interface import Keyboard

RIGHTFORWARD_GRIPPER = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
TRANSFORM_VR = np.array(
    [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
)  # geom.euler_to_rot(np.array([0, np.pi, 0]))

ENV_LOOKUP = {
    "door": DoorEnv,
    "kitchen": KitchenEnv,
    "kitchen2": Kitchen2Env,
}


def main(gui, env_type, cam_name="upview", save_video=False):
    if env_type in ENV_LOOKUP.keys():
        env_class = ENV_LOOKUP[env_type]
    else:
        env_class = EmptyEnv

    env = env_class()

    env.config["Manipulation"]["Trajectory Mode"] = "interpolation"

    if save_video:
        save_path = os.path.join(".", "{}_{}.mp4".format(env_type, cam_name))
    else:
        save_path = None
    renderer_main = CV2Renderer(
        device_id=-1, sim=env.sim, cam_name=cam_name, gui=gui, save_path=save_path, window_name='main-view',
        width=800, height=600
    )
    renderer_sub = CV2Renderer(
        device_id=-1, sim=env.sim, cam_name='robot0_replayview', gui=gui, save_path=save_path, window_name='ego-view',
        width=400, height=400
    )

    recorder = None
    # recorder = HDF5Recorder(
    #     sim=env.sim,
    #     config=env.config,
    #     file_path="./test/demo_{}_{}".format(env_type, int(time.time())),
    # )

    env.set_renderer([renderer_main, renderer_sub])
    env.set_recorder(recorder)
    env.reset(subtask=2)

    spacemouse = SpaceMouse()
    keyboard = Keyboard()
    mat_trans = np.eye(4)
    mat_trans[:3, :3] = geom.quat_to_rot([0.5, -0.5, -0.5, 0.5])  # T265
    trk_init = True

    right_pos = np.array([0.22, -0.25, 0.1])
    left_pos = np.array([0.22, 0.25, 0.1])

    # spacemouse.start()
    keyboard.start()

    done = False
    read_time = 0
    print("Press 'r' button for starting simuation.")
    while not keyboard.enable:
        pass

    while not done:

        loco_cmd = keyboard.control
        loco_val = 0
        if loco_cmd[0] > 0:
            loco_val = 1
        elif loco_cmd[0] < 0:
            loco_val = 2
        elif loco_cmd[1] > 0:
            loco_val = 3
        elif loco_cmd[1] < 0:
            loco_val = 4
        elif loco_cmd[2] > 0:
            loco_val = 5
        elif loco_cmd[2] < 0:
            loco_val = 6

        action = {}
        action["trajectory"] = {}
        action["gripper"] = {}
        action["aux"] = {}
        action["subtask"] = 0
        action["locomotion"] = loco_val

        # SpaceMouse
        inp = spacemouse.control
        rh_target_pos = np.array([inp[0], inp[1], inp[2]]) + np.array([0.22, -0.25, 0.1])
        rh_rotation_input = geom.euler_to_rot(np.array([inp[4], inp[3], inp[5]]))
        rh_target_rot = np.dot(rh_rotation_input, RIGHTFORWARD_GRIPPER)
        rh_grip = spacemouse.click
        
        lh_target_pos = left_pos
        lh_input = geom.euler_to_rot(np.array([0, 0, 0]))
        lh_grip = 0

        lh_target_rot = np.dot(lh_input, RIGHTFORWARD_GRIPPER)
        action["trajectory"]["left_pos"] = lh_target_pos
        action["trajectory"]["right_pos"] = rh_target_pos
        action["trajectory"]["right_quat"] = geom.rot_to_quat(rh_target_rot)
        action["trajectory"]["left_quat"] = geom.rot_to_quat(lh_target_rot)
        action["gripper"] = {"left": lh_grip, "right": rh_grip}

        env.step(action)

        done = not keyboard.enable

    print("Done!")
    if recorder is not None:
        recorder.close()
    t265.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", type=int, default=1, help="")
    parser.add_argument("--env", type=str, default="door", help="")
    parser.add_argument("--cam", type=str, default="upview", help="")
    parser.add_argument("--save_video", type=int, default=0, help="")
    parser.add_argument("--subtask", type=int, default=1, help="")
    args = parser.parse_args()

    gui = args.gui
    env_type = args.env
    cam_name = args.cam
    subtask = args.subtask
    save_video = args.save_video

    main(gui=gui, env_type=env_type, cam_name=cam_name, save_video=save_video)
