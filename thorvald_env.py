
import gym
from gym import spaces
import numpy as np
import math
import carb
import torch

class ThorvaldEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        skip_frame=1,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        max_episode_length=256,
        seed=0,
        headless=True,
    ) -> None:
        from omni.isaac.kit import SimulationApp

        self.headless = headless
        self._simulation_app = SimulationApp({"headless": self.headless, "anti_aliasing": 0})
        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._max_episode_length = max_episode_length
        self._steps_after_reset = int(rendering_dt / physics_dt)
        
        from omni.isaac.core import World
        from omni.isaac.wheeled_robots.robots import WheeledRobot
        from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
        from omni.isaac.core.objects import VisualCuboid
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from thorvald_kin import Thorvald, ThorvaldController
        from omni.isaac.core.articulations import ArticulationView
        



        self._my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
        self._my_world.scene.add_default_ground_plane()
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return
        
        self.thorvald = self._my_world.scene.add(Thorvald(prim_path="/World/Thorvald", name="my_thorvald",
                wheel_dof_names=[ "steering0", "steering1", "steering2", "steering3","wheel0", "wheel1", "wheel2", "wheel3"],
                wheel_dof_indices=[0,1,2,3,4,5,6,7],
                create_robot=True,
                position=torch.tensor([0, 0, 0.1])))

        self.thorvald_controller = ThorvaldController()
        self.goal = self._my_world.scene.add(
            VisualCuboid(
                prim_path="/new_cube_1",
                name="visual_cube",
                position=np.array([0.60, 0.30, 0.05]),
                size=0.5,
                color=np.array([1.0, 0, 0]),
            )
        )
        self.seed(seed)
        self.reward_range = (-float("inf"), float("inf"))
        gym.Env.__init__(self)
        self.action_space = spaces.Box(low=-1, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(16,), dtype=np.float32)

        self.max_x_velocity = 1
        self.max_y_velocity = 1
        self.max_angular_velocity = math.pi / 4
        self.reset_counter = 0
        return

    def get_dt(self):
        return self._dt

    def step(self, action):
        
        previous_thorvald_position, _ = self.thorvald.get_world_pose()
        # action forward velocity , angular velocity on [-1, 1]
        raw_x = action[0]
        raw_y = action[1]
        raw_angular = action[2]

        # we want to force the jetbot to always drive forward
        # so we transform to [0,1].  we also scale by our max velocity
        #forward = (raw_forward + 1.0) / 2.0
        #forward_velocity = forward * self.max_velocity
        x_velocity = raw_x * self.max_x_velocity
        y_velocity = raw_y * self.max_y_velocity

        # we scale the angular, but leave it on [-1,1] so the
        # jetbot can remain an ambiturner.
        angular_velocity = raw_angular * self.max_angular_velocity

        # we apply our actions to the jetbot
        for i in range(self._skip_frame):
            pos, velo = self.thorvald_controller.forward(command=[x_velocity,y_velocity, angular_velocity])
            


            from omni.isaac.core.utils.types import ArticulationAction
            self.thorvald.apply_wheel_actions(actions=ArticulationAction(joint_positions=pos, 
                                                                        joint_velocities=velo,
                                                                        joint_indices=[0,1,2,3,4,5,6,7]))
            self._my_world.step(render=False)

        observations = self.get_observations()
        info = {}
        done = False
        if self._my_world.current_time_step_index - self._steps_after_reset >= self._max_episode_length:
            done = True
        goal_world_position, _ = self.goal.get_world_pose()
        current_jetbot_position, _ = self.thorvald.get_world_pose()
        previous_dist_to_goal = np.linalg.norm(goal_world_position - previous_thorvald_position)
        current_dist_to_goal = np.linalg.norm(goal_world_position - current_jetbot_position)
        reward = previous_dist_to_goal - current_dist_to_goal
        if current_dist_to_goal < 0.1:
            done = True
        return observations, reward, done, info

    def reset(self):
        self._my_world.reset()
        self.reset_counter = 0
        # randomize goal location in circle around robot
        alpha = 2 * math.pi * np.random.rand()
        r = 1.00 * math.sqrt(np.random.rand()) + 5.20
        self.goal.set_world_pose(np.array([math.sin(alpha) * r, math.cos(alpha) * r, 0.05]))
        observations = self.get_observations()
        return observations

    def get_observations(self):
        self._my_world.render()
        jetbot_world_position, jetbot_world_orientation = self.thorvald.get_world_pose()
        jetbot_linear_velocity = self.thorvald.get_linear_velocity()
        jetbot_angular_velocity = self.thorvald.get_angular_velocity()
        goal_world_position, _ = self.goal.get_world_pose()
        return np.concatenate(
            [
                jetbot_world_position,
                jetbot_world_orientation,
                jetbot_linear_velocity,
                jetbot_angular_velocity,
                goal_world_position,
            ]
        )

    def render(self, mode="human"):
        return

    def close(self):
        self._simulation_app.close()
        return

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]
