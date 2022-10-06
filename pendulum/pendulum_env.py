import RobotDART as rd
import dartpy 
import numpy as np
import torch
import machin


class Env():

    def __init__ (
            self,
            time_step = 0.05,
            robot = 'pendulum.urdf',
            initial_positions = None,
            graphics = False
        ):
        self.time_step = time_step
        self.robot = rd.Robot(robot)
        self.initial_positions = initial_positions
        self.graphics = graphics
        self.prev_theta = 0
        self.simu = rd.RobotDARTSimu(self.time_step)

        if self.graphics:
            graphics = rd.gui.Graphics()
            self.simu.set_graphics(graphics)
            graphics.enable_shadows(enable = False, transparent = False)
            graphics.look_at([0., 2.5, 0.5], [0., 0., 0.])

        
        self.robot.fix_to_world()
        self.robot.set_actuator_types("torque")
        self.robot.set_draw_axis(self.robot.body_name(0), 0.5)
        self.robot.set_draw_axis(self.robot.body_name(1), 0.25)
        self.robot.set_positions(self.initial_positions)
        self.simu.add_robot(self.robot)
    
    def get_theta(self):
        return ((self.robot.positions()[0] + np.pi) % (2. * np.pi)) - np.pi
    
    def get_graphics(self):
        return self.graphics

    def step(self, commands):
        terminal = False
        # commands = np.clip(commands, -5., 5.)
        self.robot.set_commands(commands[0])
        self.simu.step_world()
        theta = self.get_theta().item()
        ang_vel = self.robot.velocities().item()
        reward = theta ** 2  + 0.1 * ang_vel ** 2 + 0.001 * (commands.item() ** 2)

        return np.array([np.cos(theta), np.sin(theta), ang_vel], dtype=np.float32), -reward, terminal, {}

    def reset(self):
        self.robot.reset_commands()
        self.simu.step_world()
        theta = self.get_theta().item()
        return np.array([np.cos(theta), np.sin(theta), 0])
