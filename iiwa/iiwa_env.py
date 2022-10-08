import copy
import json
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
            time_step = 0.004,
            initial_positions=None,
            graphics=False
        ):
        #Set simulation variables
        self.time_step = time_step
        self.eef_link_name = "iiwa_link_ee"
        self.simu = rd.RobotDARTSimu(self.time_step)
        self.simu.add_checkerboard_floor()
        self.robot = rd.Iiwa()
        self.robot_ghost=self.robot.clone_ghost()

        if initial_positions is not None:
            self.robot.set_positions(initial_positions)

        
        # set target joint positions
        self.target_positions = copy.copy(self.robot.positions())
        self.target_positions[0] = -2.
        self.target_positions[3] = -np.pi / 2.0
        self.target_positions[5] = np.pi / 2.0
        self.robot_ghost.set_positions(self.target_positions)
        self.tf_desired = self.robot.body_pose(self.eef_link_name)

        #Add robots to simulation
        self.simu.add_robot(self.robot)
        self.simu.add_robot(self.robot_ghost)

        #Find and save target position oin world frame  
        self.tf_desired = self.robot.body_pose(self.eef_link_name)

        #Set graphics if secected
        if graphics:
            graphics = rd.gui.Graphics()
            self.simu.set_graphics(graphics)


    def step(self, commands):
        terminal = False

        self.robot.set_commands(commands)
        for _ in range(5):
            self.simu.step_world(False)
            distance = np.abs(np.subtract(self.robot.positions(), self.robot_ghost.positions()))
            reward = 0
            for d in distance:
                reward += np.sqrt(d**2)

        return np.r_[self.robot.positions(), self.robot.body_pose(self.eef_link_name).translation()], -reward, False, {}


    def reset(self, init_pos=None, graphics=False):
        #Create new simulation
        self.simu = rd.RobotDARTSimu(self.time_step)
        self.simu.add_checkerboard_floor()

        #Add robot 
        self.robot = rd.Iiwa()
        self.robot.reset_commands()
        self.robot.set_positions(init_pos)
        self.robot.set_actuator_types("servo")
        self.simu.add_robot(self.robot)

        #Add ghost robot
        self.robot_ghost = self.robot.clone_ghost()
        self.robot_ghost.set_positions(self.target_positions)
        self.simu.add_robot(self.robot_ghost)
    

        if graphics:
            graphics = rd.gui.Graphics()
            self.simu.set_graphics(graphics)
            graphics.look_at([3., 1., 2.], [0., 0., 0.])
        
        for _ in range(5):
            self.simu.step_world()

        return np.r_[self.robot.positions(), self.robot.body_pose(self.eef_link_name).translation()]

    def confirm_env(self, run=1, json_file_path='iiwa_td3.json'):
        while True:
            try:
                data = json.loads(open(json_file_path, "r").read())
                episode = input(f'Pick episode to run: From {len(data["run_"+str(run)]["episodes"].keys())} available episodes ->  ')
                initial_positions = data[f'run_{run}']['initial_positions']
                self.reset(initial_positions, True)
                for i in data[f'run_{run}']['episodes'][str(episode)]['actions']:
                    self.step(i)
            except KeyboardInterrupt:
                break
        
    
if __name__ == '__main__':
    Env().confirm_env()