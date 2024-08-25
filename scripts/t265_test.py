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

from devices.t265 import T265
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


def main(gui, env_type, cam_name="upview", save_video=False, subtask=1):
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
    env.reset(subtask=subtask)

    t265_r = T265(img_stream=False)
    t265_l = T265(img_stream=False)
    keyboard = Keyboard(right_gripper=True, left_gripper=True)

    mat_trans_r = np.eye(4)
    mat_trans_r[:3, :3] = geom.quat_to_rot([0.5, -0.5, -0.5, 0.5])  # T265

    mat_trans_l = np.eye(4)
    mat_trans_l[:3, :3] = geom.quat_to_rot([0.5, -0.5, -0.5, 0.5])  # T265

    trk_init_r = True
    trk_init_l = True

    right_pos_offset = np.array([0.22, -0.25, 0.1])
    left_pos_offset = np.array([0.22, 0.25, 0.1])

    t265_r.start()
    t265_l.start()
    keyboard.start()

    done = False
    read_time_r = 0
    read_time_l = 0
    print("Press 'r' button for starting simuation.")
    while not keyboard.enable:
        pass

    while not done:

        if t265_r.time > read_time_r + 0.1:
            trk_pos_r = t265_r.pos
            trk_rot_r = t265_r.rot
            read_time_r += 0.1

        if t265_l.time > read_time_l + 0.1:
            trk_pos_l = t265_l.pos
            trk_rot_l = t265_l.rot
            read_time_l += 0.1

        # T265 POSE
        mat_se3_r = np.eye(4)
        mat_se3_r[:3, :3] = geom.quat_to_rot(trk_rot_r)
        mat_se3_r[:3, 3] = trk_pos_r

        if trk_init_r:
            mat_se3_base_r = np.eye(4)
            mat_se3_base_r = np.linalg.inv(mat_trans_r @ mat_se3_r) @ mat_se3_base_r
            trk_init_r = False

        trk_mat_se3_r = mat_trans_r @ mat_se3_r @ mat_se3_base_r
        pos_r = trk_mat_se3_r[:3, 3]
        quat_r = geom.rot_to_quat(trk_mat_se3_r[:3, :3])

        mat_se3_l = np.eye(4)
        mat_se3_l[:3, :3] = geom.quat_to_rot(trk_rot_l)
        mat_se3_l[:3, 3] = trk_pos_l

        if trk_init_l:
            mat_se3_base_l = np.eye(4)
            mat_se3_base_l = np.linalg.inv(mat_trans_l @ mat_se3_l) @ mat_se3_base_l
            trk_init_l = False

        trk_mat_se3_l = mat_trans_l @ mat_se3_l @ mat_se3_base_l
        pos_l = trk_mat_se3_l[:3, 3]
        quat_l = geom.rot_to_quat(trk_mat_se3_l[:3, :3])


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

        rh_target_pos = 5 * pos_r + right_pos_offset
        lh_target_pos = 5 * pos_l + left_pos_offset
        rh_input = geom.quat_to_rot(quat_r)
        lh_input = geom.quat_to_rot(quat_l)
        rh_grip, lh_grip = keyboard.grasp

        rh_target_rot = np.dot(rh_input, RIGHTFORWARD_GRIPPER)
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
    t265_r.stop()
    t265_l.stop()


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

    main(gui=gui, env_type=env_type, cam_name=cam_name, save_video=save_video, subtask=subtask)
