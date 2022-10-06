from typing import Union
import RobotDART as rd
import dartpy 
import numpy as np
import torch
import machin
import json


class Env():

    def __init__ ( self, time_step: float = 0.04):

        self.time_step = time_step
        self.robot = rd.Robot('pendulum.urdf')
        self.simu = rd.RobotDARTSimu(self.time_step)
        self.robot.fix_to_world()
        self.robot.set_actuator_types("torque")
        self.robot.set_draw_axis(self.robot.body_name(0), 0.5)
        self.robot.set_draw_axis(self.robot.body_name(1), 0.25)
        self.simu.add_robot(self.robot)
    
    def get_theta(self):
        # Get normalized angle. [-pi, pi]
        return ((self.robot.positions()[0] + np.pi) % (2. * np.pi)) - np.pi

    def step(self, commands):
        
        self.robot.set_commands(commands[0])
        self.simu.step_world()
        theta = self.get_theta().item()
        ang_vel = self.robot.velocities().item()
        forces = self.robot.forces().item()
        reward = theta ** 2  + 0.1 * ang_vel ** 2 + 0.001 * (forces ** 2)


        return np.array([np.cos(theta), np.sin(theta), ang_vel], dtype=np.float32), -reward, False, {}

    def reset(
        self, 
        initial_positions: Union[list, np.ndarray],
        render: bool = False
    ) -> np.array:

        self.simu = rd.RobotDARTSimu(self.time_step)
        self.robot = rd.Robot('pendulum.urdf')
        self.robot.fix_to_world()
        self.robot.set_actuator_types("torque")
        self.robot.set_draw_axis(self.robot.body_name(0), 0.5)
        self.robot.set_draw_axis(self.robot.body_name(1), 0.25)
        self.simu.add_robot(self.robot)

        if render:
            graphics = rd.gui.Graphics()
            self.simu.set_graphics(graphics)
            graphics.look_at([0., 2.5, 0.5], [0., 0., 0.])

        self.robot.reset_commands()
        self.robot.set_positions(initial_positions)
        self.simu.step_world()
        theta = self.get_theta().item()
        return np.array([np.cos(theta), np.sin(theta), 0])

    def confirm_env(self, run=1, json_file_path='pendulum_ppo.json'):
        while True:
            try:
                data = json.loads(open(json_file_path, "r").read())
                episode = input(f'Pick episode to run: From {len(data["run_"+str(run)]["episodes"].keys())} available episodes ->  ')
                self.reset([np.pi], True)
                for i in data[f'run_{run}']['episodes'][str(episode)]['actions']:
                    self.step(np.array([[i]]))
            except KeyboardInterrupt:
                break
        
if __name__ == '__main__':
    Env().confirm_env()
