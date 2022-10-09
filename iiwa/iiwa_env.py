import os
import copy
import json
from typing import Union
from time import sleep
import RobotDART as rd
import dartpy 
import numpy as np
import torch
import machin
from random import randrange, random


class Env():

    def __init__ (self, time_step: float = 0.004):
        #Set simulation variables
        self.time_step = time_step
        self.eef_link_name = "iiwa_link_ee"
        self.simu = rd.RobotDARTSimu(self.time_step)
        self.simu.add_checkerboard_floor()
        self.robot = rd.Iiwa()
        self.robot_ghost=self.robot.clone_ghost()

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



    def step(self, commands: Union[list, np.ndarray]) -> np.array:
        terminal = False

        self.robot.set_commands(commands)

        # Run 5 step with same action
        for _ in range(5):
            self.simu.step_world(False)
            distance = np.abs(np.subtract(self.robot.positions(), self.robot_ghost.positions()))
            reward = 0
            for d in distance:
                reward += np.sqrt(d**2)

        return np.r_[self.robot.positions(), self.robot.body_pose(self.eef_link_name).translation()], -reward, False, {}


    def reset(
        self, 
        initial_positions: Union[list, np.ndarray],
        render: bool = False
    ) -> np.array:

        #Create new simulation
        self.simu = rd.RobotDARTSimu(self.time_step)
        self.simu.add_checkerboard_floor()

        #Add robot 
        self.robot = rd.Iiwa()
        self.robot.reset_commands()
        self.robot.set_positions(initial_positions)
        self.robot.set_actuator_types("servo")
        self.simu.add_robot(self.robot)

        #Add ghost robot
        self.robot_ghost = self.robot.clone_ghost()
        self.robot_ghost.set_positions(self.target_positions)
        self.simu.add_robot(self.robot_ghost)
    

        #Set graphics if secected
        if render:
            graphics = rd.gui.Graphics()
            self.simu.set_graphics(graphics)
            graphics.look_at([3., 1., 2.], [0., 0., 0.])
        
        for _ in range(5):
            self.simu.step_world()

        return np.r_[self.robot.positions(), self.robot.body_pose(self.eef_link_name).translation()]

    def confirm_env(self, run=3, json_file_path='iiwa_ppo.json'):
        file_path = os.getcwd()
        while True:
            try:
                algorith_type = int(input("[1] -> For PPO \n[2] -> For TD3\nSelection: "))
                if algorith_type == 1:
                    json_file_path = f'{file_path}/iiwa_ppo.json'
                else:
                    json_file_path = f'{file_path}/iiwa_td3.json'

                data = json.loads(open(json_file_path, "r").read())
            except KeyboardInterrupt:
                print("\n")
                break

            while True:
                try:
                    runs_num = len(data.keys())
                    run = input(f"Select run: Available runs {runs_num}: ")
                except KeyboardInterrupt:
                    print("\n")
                    break
                while True:
                    try:
                        action_counter = 0
                        for k, v in reversed(data[f'run_{run}']['episodes'].items()):
                            if v['actions'] is not None:
                                action_counter += 1
                            else:
                                break
                        episode = input(f'Pick episode to run: From {action_counter} available episodes ->  ')
                        initial_positions = data[f'run_{run}']['initial_positions']
                        self.reset(initial_positions, True)
                        for action in data[f'run_{run}']['episodes'][str(len(data["run_"+str(run)]["episodes"].keys()) - action_counter + int(episode) + 1)]['actions']:
                            self.step(action)
                    except KeyboardInterrupt:
                        print("\n")
                        break
        
    
if __name__ == '__main__':
    Env().confirm_env()