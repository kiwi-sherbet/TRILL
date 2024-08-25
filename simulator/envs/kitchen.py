from .base import BaseEnv
from robosuite.models.arenas.table_arena import TableArena
from ..objects import PotObject, PotLidObject, StoveObject, LadleObject, TargetObject
from util import geom

import numpy as np

PLANE_MARGIN = 0.05
EDGE_MARGIN = 0.5
OBJECT_MARGIN = 0.4

# MEAN_INIT_POS = np.array([-0.5, 0.0, 0.743])
MEAN_INIT_POS = {0: np.array([-0.5, 0.0, 0.743]),
                1: np.array([0, 0, 0.743]),
                2: np.array([0, 0, 0.743])}
STD_INIT_POS = {0: np.array([0.1, 0.2, 0.0]),
                1: np.array([0.01, 0.02, 0.0]),
                2: np.array([0.02, 0.03, 0.0])}
MEAN_INIT_YAW = {0: 0.0,
                1: 0.00,
                2: 0.0
                }
STD_INIT_YAW = {0: 0.2,
                1: 0.04,
                2: 0.05}

class KitchenEnv(BaseEnv):

    def reset(self, initial_pos=None, subtask=0, **kwargs):
        
        print("kitchen reset")
        
        direction_random = 1#np.random.choice([-1, 1])
        stove_pos_random = np.random.uniform([-0.31, -0.05, 0.0], [-0.33, 0.05, 0.0], size=3) + self.table_offset
        lid_pos_random = np.random.uniform(stove_pos_random+np.array([-0.07, -direction_random*0.25-0.03, 0.0]), 
                                            stove_pos_random+np.array([ -0.04, -direction_random*0.25+0.03, 0.0]), size=3)
        if subtask == 2:
            pot_pos_random = np.random.uniform(stove_pos_random+np.array([-0.009, -0.009, 0.01]), 
                                               stove_pos_random+np.array([ -0.009, 0.009, 0.01]), size=3)
        else:
            pot_pos_random = np.random.uniform(stove_pos_random+np.array([-0.07,  direction_random*0.34-0.03, 0.0]), 
                                                stove_pos_random+np.array([ -0.04,  direction_random*0.34+0.03, 0.0]), size=3)
        self._init_object_states = {
                                    'stove': {'pos': stove_pos_random},
                                    'pot':   {'pos': pot_pos_random},
                                    'lid': {'pos': lid_pos_random},
                                    }

        if initial_pos is None:
            if subtask == 0:
                robot_pos_mean = MEAN_INIT_POS[0]
            if subtask == 1:
                robot_pos_mean = np.array([pot_pos_random[0], pot_pos_random[1], 0.743]) + np.array([-0.37, 0.0, 0.0])
            else:
                robot_pos_mean = np.array([stove_pos_random[0], stove_pos_random[1], 0.743]) + np.array([-0.43, 0.0, 0.0])
            self._init_robot_states = {'pos': np.random.normal(robot_pos_mean, STD_INIT_POS[subtask], size=3),
                                       'yaw': np.random.normal(MEAN_INIT_YAW[subtask], STD_INIT_YAW[subtask])}
        else:
            self._init_robot_states = initial_pos

        out = super().reset(initial_pos=self._init_robot_states, subtask=subtask, **kwargs)
        return out


    def _load_model(self):
        
        super()._load_model()

        # Create an environment
        self.table_offset = np.array((0.65, 0, 0.7))
        self.table_size = np.array((1.0, 2.0, 0.05))
        self.arena = TableArena(table_full_size=self.table_size, table_offset=self.table_offset, has_legs=True)

        # initialize objects of interest
        self.objects = {
            'pot': PotObject(name="pot"),
            'lid': PotLidObject(name="lid"),
            'stove': StoveObject(name="stove"),
        }

        self.world.merge(self.arena)
        for object in self.objects.values():
            self.world.merge_assets(object)
            self.world.worldbody.append(object.get_obj())

    def _reset_objects(self):
        
        if '_init_object_states' in self.__dict__:

            for name, object in self.objects.items():
                joint_id = self.sim.model.joint_name2id(object.naming_prefix+'joint0')
                joint_qposadr = self.sim.model.jnt_qposadr[joint_id]
                if 'pos' in self._init_object_states[name]:
                    self.sim.data.qpos[joint_qposadr:joint_qposadr +3]\
                                = np.copy(self._init_object_states[name]['pos'])
                if 'quat' in self._init_object_states[name]:
                    self.sim.data.qpos[joint_qposadr+3:joint_qposadr +7]\
                                = np.copy(self._init_object_states[name]['quat'])


class Kitchen2Env(BaseEnv):


    def reset(self, initial_pos=None, subtask=0, **kwargs):

        print("kitchen2 reset", subtask)
        
        direction_random = 0#np.random.choice([-1, 1])
        stove_pos_random = self.table_offset + np.random.uniform(np.array([-0.25*self.table_size[0], -direction_random*0.6-0.12, 0.01]), 
                                                                np.array([-0.25*self.table_size[0], -direction_random*0.6+0.12, 0.01]), size=3) # random distribution of toolbox (relative to hammer position)
        stove_quat_random = np.array([1.0, 0.0, 0.0, 0.0])
        pot_pos_random = stove_pos_random + np.array([0, 0, 0.01]) # random distribution of toolbox (relative to hammer position)
        pot_quat_random = np.array([1.0, 0.0, 0.0, 0.0])

        if subtask == 2:
            ladle_pos_random = stove_pos_random + np.random.uniform([-0.02, -0.05, 0.2],
                                                                    [ 0.02,  0.05, 0.2], size=3)
            ladle_quat_random = geom.euler_to_quat([np.random.uniform(-np.pi/6, np.pi/6), np.pi, np.random.uniform(-np.pi/2, np.pi/2)])[[3, 0, 1, 2]]
            # target_pos_random = stove_pos_random + np.random.uniform(np.array([0.05, -0.15, 0.25]), np.array([0.15,  0.15, 0.35]), size=3)
        else:
            ladle_pos_random = self.table_offset + np.random.uniform([-0.25*self.table_size[0], -0.02, 0.15], # random distribution of hammer
                                                                [-0.25*self.table_size[0],  0.02, 0.15], size=3)
            ladle_quat_random = geom.euler_to_quat([0, np.pi, np.random.uniform(-np.pi/2, np.pi/2)])[[3, 0, 1, 2]]
            # target_pos_random = np.array([0.0, 0.0, -1.0]) # dummy

        self._init_object_states = {
                                    'stove': {'pos': stove_pos_random,
                                              'quat': stove_quat_random},
                                    'pot':   {'pos': pot_pos_random,
                                              'quat': pot_quat_random},
                                    'ladle': {'pos': ladle_pos_random,
                                              'quat': ladle_quat_random},
                                    # 'target': {'pos': target_pos_random},
                                    }

        if initial_pos is None:
            if subtask == 0:
                robot_pos_mean = MEAN_INIT_POS[0]
            elif subtask == 1:
                robot_pos_mean = np.array(
                    [ladle_pos_random[0], ladle_pos_random[1] - direction_random * 0.2, 0.743]) + np.array([-0.4, 0.0, 0.0]) ##<< If you want to change the robot position, change this, such as line np.array([-0.48, 0.0, 0.0]
            elif subtask == 2:
                robot_pos_mean = np.array(
                    [stove_pos_random[0], stove_pos_random[1], 0.743]) + np.array([-0.4, 0.0, 0.0])
            self._init_robot_states = {'pos': np.random.normal(robot_pos_mean, STD_INIT_POS[subtask], size=3),
                                        'yaw': np.random.normal(MEAN_INIT_YAW[subtask], STD_INIT_YAW[subtask])}
        else:
            self._init_robot_states = initial_pos

        out = super().reset(initial_pos=self._init_robot_states, subtask=subtask, **kwargs)
        return out

    def _load_model(self):

        super()._load_model()

        # Create an environment
        self.table_offset = np.array((0.75, 0, 0.625))
        self.table_size = np.array((0.75, 2.0, 0.05))

        self.arena = TableArena(
            table_full_size=self.table_size, table_offset=self.table_offset, has_legs=True)
        # initialize objects of interest
        self.objects = {
            'pot': PotObject(name="pot"),
            'ladle': LadleObject(name="ladle"), 
            'stove': StoveObject(name="stove"),
        }
        # self.target = TargetObject(name="target")

        self.world.merge(self.arena)
        # self.world.merge_assets(self.target)
        # self.world.worldbody.append(self.target.get_obj())
        for object in self.objects.values():
            self.world.merge_assets(object)
            self.world.worldbody.append(object.get_obj())

    def _reset_objects(self):

        if '_init_object_states' in self.__dict__:

            # target_body_id = self.sim.model.body_name2id(self.target.root_body)
            # self.sim.model.body_pos[target_body_id] = np.copy(self._init_object_states['target']['pos'])

            for name, object in self.objects.items():
                joint_id = self.sim.model.joint_name2id(object.naming_prefix+'joint0')
                joint_qposadr = self.sim.model.jnt_qposadr[joint_id]
                if 'pos' in self._init_object_states[name]:
                    self.sim.data.qpos[joint_qposadr:joint_qposadr +3]\
                                = np.copy(self._init_object_states[name]['pos'])
                if 'quat' in self._init_object_states[name]:
                    self.sim.data.qpos[joint_qposadr+3:joint_qposadr +7]\
                                = np.copy(self._init_object_states[name]['quat'])

        # joint_id = self.sim.model.joint_name2id('pot_joint0')
        # joint_qposadr = self.sim.model.jnt_qposadr[joint_id]

        # target_body_id = self.sim.model.body_name2id(self.target.root_body)
        # self.sim.model.body_pos[target_body_id] = np.array([0.55, 0.6, 0.9])