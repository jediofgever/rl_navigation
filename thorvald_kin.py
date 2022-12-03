from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.controllers import BaseController
from typing import Optional
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.wheeled_robots.robots import WheeledRobot

import numpy as np
from math import pow, sqrt, cos, sin, atan2, fabs, fmod
import math

class ThorvaldController(BaseController):
    
    def __init__(self):
        super().__init__(name="thorvald_controller")
        # An open loop controller that uses a unicycle model
        self._wheel_radius = 0.2
        self._robot_length = -0.67792
        self._y_offset = -0.75
        self._wheel_base = 0.1125
        return

    def normalize_angle(self, ang):
        result = fmod(ang + math.pi, 2.0*math.pi)
        if(result <= 0.0): 
            return result + math.pi
        return result - math.pi
  
    def forward(self, command):
        # command will have two elements, first element is the forward velocity
        # second element is the angular velocity (yaw only).
        steering = [0.0, 0.0, 0.0, 0.0]
        speed = [0.0, 0.0, 0.0, 0.0]

        vx = command[0]
        vy = command[1]
        wz = command[2]

        if (wz != 0.0):
            turn_rad_d = sqrt(pow(vx,2) + pow(vy,2)) / wz
            turn_rad_ang = atan2(vy,vx)
            turn_rad_x = - turn_rad_d * sin(turn_rad_ang)
            turn_rad_y = turn_rad_d * cos(turn_rad_ang)

            drive_x = -0.67
            drive_y = -0.75
            steering[0] = self.normalize_angle(-atan2((turn_rad_x - drive_x), (turn_rad_y - drive_y)) + math.pi * (wz < 0))
            speed[0] = sqrt(pow(turn_rad_x - drive_x, 2) + pow(turn_rad_y - drive_y, 2)) * fabs(wz)

            drive_x = 0.67
            drive_y = -0.75
            steering[1] = self.normalize_angle(-atan2((turn_rad_x - drive_x), (turn_rad_y - drive_y)) + math.pi * (wz < 0))
            speed[1] = sqrt(pow(turn_rad_x - drive_x, 2) + pow(turn_rad_y - drive_y, 2)) * fabs(wz)

            drive_x = 0.67
            drive_y = 0.75
            steering[2] = self.normalize_angle(-atan2((turn_rad_x - drive_x), (turn_rad_y - drive_y)) + math.pi * (wz < 0))
            speed[2] = sqrt(pow(turn_rad_x - drive_x, 2) + pow(turn_rad_y - drive_y, 2)) * fabs(wz)

            drive_x = -0.67
            drive_y = 0.75
            steering[3] = self.normalize_angle(-atan2((turn_rad_x - drive_x), (turn_rad_y - drive_y)) + math.pi * (wz < 0))
            speed[3] = sqrt(pow(turn_rad_x - drive_x, 2) + pow(turn_rad_y - drive_y, 2)) * fabs(wz)
        else:
            for i in range(0,4):            
                steering[i] = atan2(vy,vx)  
                speed[i] = sqrt(pow(vx,2) + pow(vy,2)) / self._wheel_radius


        joint_velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        joint_positions[0:4] = steering
        joint_velocities[4:8] = speed

        # A controller has to return an ArticulationAction
        #return ArticulationAction(joint_positions=joint_positions, joint_velocities=joint_velocities)   
        return joint_positions, joint_velocities 

class Thorvald(WheeledRobot):
    def __init__(
        self,
        prim_path: str,
        wheel_dof_names: Optional[str] = None,
        wheel_dof_indices: Optional[int] = None,
        name: str = "wheeled_robot",
        usd_path: Optional[str] = None,
        create_robot: Optional[bool] = False,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:

        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
            self._usd_path = "/home/atas/Documents/thorvald.usd"

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            wheel_dof_indices=wheel_dof_indices,
            wheel_dof_names=wheel_dof_names,
            name=name,
            usd_path=usd_path,
            position=position,
            orientation=orientation,
            create_robot=False
        )
