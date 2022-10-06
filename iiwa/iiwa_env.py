import copy
import re
from time import sleep
import RobotDART as rd
import dartpy 
import numpy as np
import torch
import machin
from random import randrange, random


class Env():

    def __init__ (
            self,
            time_step = 0.005,
            robot = 'iiwa',
            initial_positions = None,
            graphics = False
        ):
        self.time_step = time_step
        self.robot = rd.Iiwa()
        self.initial_positions = initial_positions
        self._sum_error = 0
        self.graphics = graphics
        self.eef_link_name = "iiwa_link_ee"
        self.simu = rd.RobotDARTSimu(self.time_step)
        self.position_limits = [2.96705973, 2.0943951,  2.96705973, 2.0943951,  2.96705973, 2.0943951, 3.05432619]
        self.action_range = np.array([1.48352986, 1.48352986, 1.74532925, 1.30899694, 2.26892803, 2.35619449, 2.35619449])

        if self.graphics:
            config = rd.gui.GraphicsConfiguration()
            config.shadowed = False
            config.transparent_shadows = False
            graphics = rd.gui.Graphics(config)
            #graphics.enable_shadows(confgi)
            self.simu.set_graphics(graphics)
            graphics.look_at([3., 1., 2.], [0., 0., 0.])

        
        # self.robot.fix_to_world()
        self.robot.set_actuator_types("servo")
        self.robot_ghost = self.robot.clone_ghost()

        # set initial joint positions
        self.target_positions = copy.copy(self.robot.positions())
        self.target_positions[0] = -2.
        self.target_positions[3] = -np.pi / 2.0
        self.target_positions[5] = np.pi / 2.0
        self.robot.set_positions(self.target_positions)

        # get end-effector pose
        eef_link_name = "iiwa_link_ee"
        self.tf_desired = self.robot.body_pose(eef_link_name)

        # set robot to random positions
        self.robot.set_positions(self.initial_positions)

        self.simu.add_robot(self.robot)
        self.robot_ghost.set_positions(self.target_positions)
        self.simu.add_robot(self.robot_ghost)
        self.simu.add_checkerboard_floor()


    
    def get_graphics(self):
        return self.graphics

    def step(self, commands):
        terminal = False

        # commands  = np.clip(commands, -1*self.action_range, self.action_range) 
        for _ in range(5):
            self.robot.set_commands(commands)
            self.simu.step_world(False, False)
        # Get each joints distance.

        distance = np.abs(np.subtract(self.robot.positions(), self.robot_ghost.positions()))
        eef_distance = np.linalg.norm((self.robot.body_pose(self.eef_link_name).translation() - self.robot_ghost.body_pose(self.eef_link_name).translation()))

        if (all( i < 0.02 for i in distance)):
            print("DONE")
            terminal = True

        reward = 0
        reward += sum(np.sqrt(distance**2))
        # reward = 0.7 * reward + .3 * eef_distance

        return np.r_[self.robot.positions()], -reward, terminal, {}


    def reset(self):
        self.robot.reset_commands()
        self.robot.set_positions(self.initial_positions)
        # self.target_positions[0] = -1.95
        # self.target_positions[3] = -np.pi / 2.1
        # self.target_positions[5] = np.pi / 2.0
        # self.robot.set_positions(self.target_positions)
        for _ in range(5):
            self.simu.step_world(True, False)
        return np.r_[self.robot.positions()]

# env = Env(graphics=True)
# while(1):
#     env.reset()
