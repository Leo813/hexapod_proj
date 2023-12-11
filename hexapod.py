# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch
from datetime import datetime

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.base.vec_task import VecTask

from typing import Tuple, Dict
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from isaacgym.terrain_utils import *

#plt.figure(figsize=(8, 4), layout='constrained')
# plt.plot(x, x, label='linear')  # Plot some data on the (implicit) axes.
# plt.plot(x, x**2, label='quadratic')  # etc.
# plt.plot(x, x**3, label='cubic')
# plt.xlabel('x label')
# plt.ylabel('y label')
# plt.title("Simple Plot")
# plt.legend()

writer = SummaryWriter()

class Anymal10(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # plane params
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang

        self.base_init_state = state

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        self.cfg["env"]["numObservations"] = 104 #69 # 0923
        self.cfg["env"]["numActions"] = 18 # 0628 leo

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # other
        self.dt = self.sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]
        print(self.dt)
        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)
        ##
        # _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # self.rb_states = gymtorch.wrap_tensor(_rb_states).view(self.num_envs, 19, 13)

        ### leo force sensor ###
        self.sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        #print(self.sensor_tensor)
        vec_sensor_tensor = gymtorch.wrap_tensor(self.sensor_tensor).view(self.num_envs, 6)
        self.sensor_forces = vec_sensor_tensor[..., 0:3]
        self.sensor_torques = vec_sensor_tensor[..., 3:6]
        ########################

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        ####################### leo #####################
        actors_per_env = 4
        self.dof_state_tensor2 = dof_state_tensor
        vec_root_tensor = gymtorch.wrap_tensor(actor_root_state).view(self.num_envs, actors_per_env, 13) # leo
        self.root_states = vec_root_tensor[..., 0, 0:13]
        self.cube_states = vec_root_tensor[..., 1, 0:13]
        # other 2 box
        self.cube_states2 = vec_root_tensor[..., 2, 0:13]
        self.cube_states3 = vec_root_tensor[..., 3, 0:13]
        ###################################################
        #self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        #print(self.dof_state.shape)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)

        self.commands = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        # self.commands_y = self.commands.view(self.num_envs, 3)[..., 1]
        # self.commands_x = self.commands.view(self.num_envs, 3)[..., 0]
        # self.commands_yaw = self.commands.view(self.num_envs, 3)[..., 2]
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        # initialize some data used later on
        self.extras = {}
        self.initial_actors_states = gymtorch.wrap_tensor(actor_root_state).clone() # leo
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # actions buffer
        self.dof_position_targets = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device, requires_grad=False)
        #self.actions_buf = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        ################################## leo for cube #####################################
        self.all_actors_indices = torch.arange(actors_per_env * self.num_envs, dtype=torch.int32, device=self.device).view(self.num_envs, actors_per_env)
        self.all_bbot_indices = actors_per_env * torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        #########################################################################################################
        base_quat = self.root_states[:, 3:7]
        self.base_ang_vel = quat_rotate_inverse(base_quat, self.root_states[:, 10:13])
        self.last_ang_vel = torch.zeros_like(self.base_ang_vel)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        ### height stuff
        self.height_points = self.init_height_points()
        self.measured_heights = None
        #self.height_samples = None

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # leo my stuff
        self.offset_z = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.offset_z[:, 2] = 0.1

        self.count_num = 0
        self.prog = 0

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        # self._create_ground_plane()
        self.height_samples = self._terrain_create()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _terrain_create(self):
        # create all available terrain types
        num_terains = 1 #8
        terrain_width = 100.0
        terrain_length = 100.0
        horizontal_scale = 0.1 #0.04  # [m]
        vertical_scale = 0.005 #0.004  # [m]
        num_rows = int(terrain_width / horizontal_scale)
        num_cols = int(terrain_length / horizontal_scale)
        heightfield = np.zeros((num_terains * num_rows, num_cols), dtype=np.int16)

        def new_sub_terrain(): return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale,
                                                 horizontal_scale=horizontal_scale)
        print(heightfield.shape)
        heightfield[0:num_rows, :] = random_uniform_terrain(new_sub_terrain(), min_height=-0.01, max_height=0.01,
                                                            step=0.005, downsampled_scale=0.5).height_field_raw
        # heightfield[num_rows:2 * num_rows, :] = sloped_terrain(new_sub_terrain(), slope=-0.5).height_field_raw
        # heightfield[2 * num_rows:3 * num_rows, :] = pyramid_sloped_terrain(new_sub_terrain(),
        #                                                                    slope=-0.5).height_field_raw
        # heightfield[0:num_rows, :] = discrete_obstacles_terrain(new_sub_terrain(), max_height=0.05,
        #                                                                        min_size=0.5, max_size=2.,
        #                                                                        num_rects=1000).height_field_raw
        # heightfield[0:num_rows, :] = wave_terrain(new_sub_terrain(), num_waves=3.,
        #                                                          amplitude=1.).height_field_raw # pos: [23.0, 19.0, -0.85], spacing=0.01        # heightfield[5 * num_rows:6 * num_rows, :] = stairs_terrain(new_sub_terrain(), step_width=0.75,
        #                                                            step_height=-0.5).height_field_raw
        # heightfield[6 * num_rows:7 * num_rows, :] = pyramid_stairs_terrain(new_sub_terrain(), step_width=0.75,
        #                                                                    step_height=-0.5).height_field_raw
        # heightfield[7 * num_rows:8 * num_rows, :] = stepping_stones_terrain(new_sub_terrain(), stone_size=1.,
        #                                                                     stone_distance=1., max_height=0.5,
        #                                                                     platform_size=0.).height_field_raw

        # add the terrain as a triangle mesh
        vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale,
                                                             vertical_scale=vertical_scale, slope_threshold=1.5)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = vertices.shape[0]
        tm_params.nb_triangles = triangles.shape[0]
        tm_params.transform.p.x = -1.
        tm_params.transform.p.y = -1.
        self.gym.add_triangle_mesh(self.sim, vertices.flatten(), triangles.flatten(), tm_params)

        self.height_samples = torch.tensor(heightfield).view(num_rows, num_cols).to(self.device)

        return self.height_samples

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/hexapod2/urdf/test_spider6.urdf" #asset_file = "urdf/anymal_c/urdf/anymal.urdf"
        #asset_path = os.path.join(asset_root, asset_file)
        #asset_root = os.path.dirname(asset_path)
        #asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = False # 0704 leo
        asset_options.fix_base_link = False #self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False
        asset_options.use_mesh_materials = True # 0704 leo

        anymal_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(anymal_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(anymal_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        body_names = self.gym.get_asset_rigid_body_names(anymal_asset)

        self.dof_names = self.gym.get_asset_dof_names(anymal_asset)
        # 0704 leo
        #extremity_name = "SHANK" if asset_options.collapse_fixed_joints else "FOOT"
        feet_names = [s for s in body_names if "tibia" in s] # 3 6 9 12 15 18
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        knee_names = [s for s in body_names if "femur" in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        # 0704 leo
        thigh_names = [s for s in body_names if "coxa" in s]
        self.thigh_indices = torch.zeros(len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(anymal_asset)
        # leo
        self.bbot_dof_lower_limits = []
        self.bbot_dof_upper_limits = []
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            #dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
            dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"] #self.Kp
            dof_props['damping'][i] = self.cfg["env"]["control"]["damping"] #self.Kd
            ### leo
            #dof_props['velocity'][i] = 1.0
            self.bbot_dof_lower_limits.append(dof_props['lower'][i])
            self.bbot_dof_upper_limits.append(dof_props['upper'][i])

        #leo
        self.bbot_dof_lower_limits = to_torch(self.bbot_dof_lower_limits, device=self.device)
        self.bbot_dof_upper_limits = to_torch(self.bbot_dof_upper_limits, device=self.device)
        spacing = 1.
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.anymal_handles = []
        self.envs = []

        # create force sensors attached to the tray body
        #box1_idx = self.gym.find_asset_rigid_body_index(anymal_asset, "box1")
        sensor_pose = gymapi.Transform()
        self.gym.create_asset_force_sensor(anymal_asset, 5, sensor_pose)
        ########################### load ##############################
        # create ball asset
        self.cube_length = 0.17
        cube_options = gymapi.AssetOptions()
        cube_options.density = 30.0 # 125 #100 # 20 for 3
        cube_asset = self.gym.create_box(self.sim, self.cube_length, self.cube_length, 0.1, cube_options) #0.02
        ###
        # self.ball_radius = 0.02
        # ball_options = gymapi.AssetOptions()
        # ball_options.density = 500
        # cube_asset = self.gym.create_sphere(self.sim, self.ball_radius, ball_options)
        ##############################################################

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            anymal_handle = self.gym.create_actor(env_ptr, anymal_asset, start_pose, "anymal", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, anymal_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, anymal_handle)
            self.envs.append(env_ptr)
            self.anymal_handles.append(anymal_handle)

            ####################### cube #######################
            cube_pose = gymapi.Transform()
            cube_pose.p.x = -0.05+6
            cube_pose.p.y = 0.1+6 # 0.1
            cube_pose.p.z = 0.30+0.01
            cube_handle = self.gym.create_actor(env_ptr, cube_asset, cube_pose, "cube1", i, 0, 0)
            # other 2 box
            cube_pose.p.x = -0.05+6
            cube_pose.p.y = 0.1+6 # 0.1
            cube_pose.p.z = 0.4+0.01
            cube_handle2 = self.gym.create_actor(env_ptr, cube_asset, cube_pose, "cube2", i, 0, 0)
            cube_pose.p.x = -0.05+6
            cube_pose.p.y = 0.1+6 # 0.1
            cube_pose.p.z = 0.5+0.01
            cube_handle3 = self.gym.create_actor(env_ptr, cube_asset, cube_pose, "cube3", i, 0, 0)
            ##

            # set ball restitution
            props_shape = self.gym.get_actor_rigid_shape_properties(env_ptr, cube_handle)
            props_shape[0].friction = 1.4
            #props_shape[0].rolling_friction = 1.0
            #props_shape[0].torsion_friction = 1.0
            #props_shape[0].compliance = 30.0
            #props_shape[0].restitution = 0.75
            self.gym.set_actor_rigid_shape_properties(env_ptr, cube_handle, props_shape)
            props_shape = self.gym.get_actor_rigid_shape_properties(env_ptr, cube_handle2)
            props_shape[0].friction = 1.4
            #props_shape[0].compliance = 30.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, cube_handle2, props_shape)
            props_shape = self.gym.get_actor_rigid_shape_properties(env_ptr, cube_handle3)
            props_shape[0].friction = 1.4
            #props_shape[0].compliance = 30.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, cube_handle3, props_shape)

            # # set plate restitution
            # props_plate = self.gym.get_actor_rigid_shape_properties(env_ptr, anymal_handle)#
            # props_plate[0].friction = 1.2
            # #props_plate[0].rolling_friction = 1.0
            # #props_plate[0].torsion_friction = 1.0
            # props_plate[0].compliance = 0.0
            # #props_plate[0].restitution = 0.75
            # self.gym.set_actor_rigid_shape_properties(env_ptr, anymal_handle, props_plate)

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], knee_names[i])
        # 0704 leo
        for i in range(len(thigh_names)):
            self.thigh_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], thigh_names[i])

        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], "base_link")

    def pre_physics_step(self, actions):
        # leo
        actions[:, 2] = -actions[:, 1]
        actions[:, 5] = -actions[:, 4]
        actions[:, 8] = -actions[:, 7]
        actions[:, 11] = -actions[:, 10]
        actions[:, 14] = -actions[:, 13]
        actions[:, 17] = -actions[:, 16]

        self.actions = actions.clone().to(self.device)

        self.dof_position_targets += self.dt * 20 * self.actions
        self.dof_position_targets[:] = tensor_clamp(self.dof_position_targets, self.bbot_dof_lower_limits, self.bbot_dof_upper_limits)
        self.dof_position_targets[:, 2] = -self.dof_position_targets[:, 1]
        self.dof_position_targets[:, 5] = -self.dof_position_targets[:, 4]
        self.dof_position_targets[:, 8] = -self.dof_position_targets[:, 7]
        self.dof_position_targets[:, 11] = -self.dof_position_targets[:, 10]
        self.dof_position_targets[:, 14] = -self.dof_position_targets[:, 13]
        self.dof_position_targets[:, 17] = -self.dof_position_targets[:, 16]
        targets = self.dof_position_targets * self.action_scale
        #targets = self.action_scale * self.actions # + self.default_dof_pos

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))
        #self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        self.last_ang_vel[:] = self.base_ang_vel[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_actions[:] = self.actions[:]

    def init_height_points(self):
        # 1mx1.6m rectangle (without center line)
        y = 0.02 * torch.tensor([-12, -11, -10, -9, -8, -7, 7, 8, 9, 10, 11, 12], device=self.device, requires_grad=False)  # 10-50cm on each side
        x = 0.02 * torch.tensor([-12, -11, -10, -9, -8, -7, 7, 8, 9, 10, 11, 12], device=self.device, requires_grad=False)  # 20-80cm on each side
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()

        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def get_heights(self, env_ids=None):
        # if self.cfg["env"]["terrain"]["terrainType"] == 'plane':
        #     return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        # elif self.cfg["env"]["terrain"]["terrainType"] == 'none':
        #     raise NameError("Can't measure height with terrain type 'none'")

        # prepare quantities
        base_quat = self.root_states[:, 3:7]

        if env_ids:
            points = quat_apply_yaw(base_quat[env_ids].repeat(1, self.num_height_points),
                                    self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(base_quat.repeat(1, self.num_height_points), self.height_points) + (
            self.root_states[:, :3]).unsqueeze(1)

        #points += self.terrain.border_size
        points = (points / 0.02).long() # horizontal_scale = 0.02
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0])
        py = torch.clip(py, 0, self.height_samples.shape[1])

        heights1 = self.height_samples[px, py]

        heights2 = self.height_samples[px + 1, py + 1]
        heights = torch.min(heights1, heights2)

        return heights.view(self.num_envs, -1) * 0.01 # self.terrain.vertical_scale = 0.01

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_anymal_reward(
            # tensors
            self.root_states,
            self.last_ang_vel,
            self.dof_vel,
            self.last_dof_vel,
            actions,
            self.last_actions,
            self.commands,
            self.torques,
            self.contact_forces,
            #self.feet_indices,
            self.progress_buf,
            # Dict
            self.rew_scales,
            # other
            self.base_index,
            self.max_episode_length,
            self.count_num,
            self.offset_z, # leo
            # self.cube_states,
            # self.cube_states2,
            # self.cube_states3
        )

        base_quat = self.root_states[:, 3:7]
        base_lin_vel = quat_rotate_inverse(base_quat, self.root_states[:, 7:10])
        base_ang_vel = quat_rotate_inverse(base_quat, self.root_states[:, 10:13])

        # for i in range(self.num_envs):
        #     writer.add_scalar('v'+str(i), base_lin_vel[i, 1], self.count_num)
        self.count_num += 1
        #writer.close()
        #print(self.count_num)
        #self.actions_buf = actions

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim) # leo
        self.gym.refresh_rigid_body_state_tensor(self.sim)


        #input() #leo 0903
        #self.measured_heights = self.get_heights()
        #heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.0 - self.measured_heights, -1, 1.) * 5.0 #self.height_meas_scale=5.0

        self.obs_buf[:] = compute_anymal_observations(  # tensors
                                                        self.root_states,
                                                        self.commands,
                                                        self.sensor_forces,
                                                        self.dof_pos,
                                                        self.default_dof_pos,
                                                        self.dof_vel,
                                                        self.gravity_vec,
                                                        self.torques,
                                                        self.contact_forces,
                                                        self.actions,
                                                        # scales
                                                        self.lin_vel_scale,
                                                        self.ang_vel_scale,
                                                        self.dof_pos_scale,
                                                        self.dof_vel_scale
        )

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        actor_indices = self.all_actors_indices[env_ids].flatten() # leo
        #print(actor_indices.shape)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     #gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(self.initial_actors_states),
                                                     #gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
                                                     gymtorch.unwrap_tensor(actor_indices), len(actor_indices))
        #print(self.dof_pos[env_ids])
        #self.gym.set_dof_state_tensor_indexed(self.sim,
        #                                      gymtorch.unwrap_tensor(self.dof_state),
        #                                      gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # leo 0918
        bbot_indices = self.all_bbot_indices[env_ids].flatten()
        #self.dof_states[env_ids] = self.initial_dof_states[env_ids]
        self.gym.set_dof_state_tensor_indexed(self.sim, self.dof_state_tensor2, gymtorch.unwrap_tensor(bbot_indices),
                                              len(bbot_indices))

        # self.commands_x[env_ids] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        # self.commands_y[env_ids] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        # self.commands_yaw[env_ids] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()
        ###
        self.commands[env_ids, 2] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids, 1] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1),device=self.device).squeeze()

        self.commands[env_ids, 1] *= (torch.sqrt(torch.square(self.commands[env_ids, 1])) > 0.3)
        self.commands[env_ids, 1] = torch.where(self.commands[env_ids, 1] > 0.3, self.commands[env_ids, 1] / self.commands[env_ids, 1] * 1.0, self.commands[env_ids, 1])
        self.commands[env_ids, 1] = torch.where(self.commands[env_ids, 1] < -0.3, self.commands[env_ids, 1] / self.commands[env_ids, 1] * -1.0, self.commands[env_ids, 1])

        self.commands[env_ids, 2] *= (torch.sqrt(torch.square(self.commands[env_ids, 2])) > 0.3)
        self.commands[env_ids, 2] = torch.where(self.commands[env_ids, 2] > 0.3, self.commands[env_ids, 2] / self.commands[env_ids, 2] * 1.0, self.commands[env_ids, 2])
        self.commands[env_ids, 2] = torch.where(self.commands[env_ids, 2] < -0.3, self.commands[env_ids, 2] / self.commands[env_ids, 2] * -1.0, self.commands[env_ids, 2])

        self.commands[env_ids, 0] = torch.where(self.commands[env_ids, 1] == 0., -1.0, 0.)
        self.commands[env_ids, 3] = torch.where(self.commands[env_ids, 2] == 0., -1.0, 0.)

        self.commands[env_ids, 0] = 0.
        self.commands[env_ids, 1] = 1.
        self.commands[env_ids, 2] = 0.
        self.commands[env_ids, 3] = -1.
        # self.commands[:, 1] = torch.where(self.commands[:, 1] >= 0, (self.commands[:, 1] / self.commands[:, 1]), (self.commands[:, 1] / self.commands[:, 1]) * (-1.0))

        # self.commands[env_ids] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)  # set small commands to zero
        # self.commands[env_ids] *= (self.commands[env_ids, 2] > 0.5).unsqueeze(1)  # set small commands to zero

        self.last_ang_vel[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_actions[env_ids] = 0.

        # actions buffer
        self.dof_position_targets[env_ids] = torch.tensor([0., 1., -1., 0., 1., -1., 0., 1., -1., 0., 1., -1., 0., 1., -1., 0., 1., -1.], dtype=torch.float32, device=self.device)
        #self.action_buf = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_anymal_reward(
    # tensors
    root_states,
    last_ang_vel,
    dof_vel,
    last_dof_vel,
    actions,
    last_actions,
    commands,
    torques,
    contact_forces,
    #knee_indices,
    episode_lengths,
    # Dict
    rew_scales,
    # other
    base_index,
    max_episode_length,
    count,
    offset_z, #leo
    # cube_states,
    # cube_states2,
    # cube_states3
):
    # (reward, reset, feet_in air, feet_air_time, episode sums)
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], int, int, int, Tensor) -> Tuple[Tensor, Tensor] # leo 9 => 12

    # prepare quantities (TODO: return from obs ?)
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10])
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])

    # my rewards

    base_pos = root_states[:, 0:3]
    robot_x, robot_y, robot_z = get_euler_xyz(base_quat)

    distance = torch.sqrt(base_pos[:, 0]*base_pos[:, 0] + base_pos[:, 1]*base_pos[:, 1])
    distance_err = torch.exp(-8 * distance)

    # velocity tracking reward
    # lin_vel_error = torch.square(25 * (commands[:, 1] - base_lin_vel[:, 1]))#torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
    # ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
    # rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * 1.0
    # rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * 0.5
    # print('a', commands)
    vy = commands[:, 1] * base_lin_vel[:, 1]
    lin_rw = torch.tanh(8 * vy) * 2.5
    vx = torch.sqrt(torch.square(base_lin_vel[:, 0]))
    lin_x_err = (torch.tanh(25 * vx - 3) + 1.0) * 0.5

    vang = commands[:, 2] * base_ang_vel[:, 2]
    ang_rw = torch.tanh(2*vang) * 2.0

    lin_pun = commands[:, 0] * torch.sqrt(torch.square(base_lin_vel[:, 1])) * ang_rw * 0.
    ang_pun = commands[:, 3] * torch.tanh(torch.sqrt(torch.square(last_ang_vel[:, 2] + base_ang_vel[:, 2]))) * 1.0 * lin_rw * 0.25

    # other base velocity penalties
    # rew_lin_vel_z = torch.square(base_lin_vel[:, 2]) * -0.1
    # rew_ang_vel_xy = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1) * -0.05

    # torque penalty
    rew_torque = torch.sum(torch.square(torques), dim=1) * rew_scales["torque"]

    # joint acc penalty
    rew_joint_acc = torch.sum(torch.square(last_dof_vel - dof_vel), dim=1) * -0.002

    # action rate penalty
    rew_action_rate = torch.sum(torch.square(last_actions - actions), dim=1) * -0.02

    # angv acc penalty
    #rew_angv_acc = torch.square(last_ang_vel[:, 2] - base_ang_vel[:, 2]) * -0.5

    # cube_h = cube_states[:, 2]
    # cube_posXY = cube_states[:, 0:2]
    # cube_posX = cube_posXY[:, 0] + 0.05
    # cube_posY = cube_posXY[:, 1] - 0.1
    # cube_pos_err = torch.sqrt((cube_posX-base_pos[:, 0])*(cube_posX-base_pos[:, 0]) + (cube_posY-base_pos[:, 1])*(cube_posY-base_pos[:, 1]))
    # cube_pos_err = 0.5 * torch.tanh(25*cube_pos_err-3) + 0.495
    # cube_h2 = cube_states2[:, 2]
    # cube_posXY2 = cube_states2[:, 0:2]
    # cube_posX2 = cube_posXY2[:, 0] + 0.05
    # cube_posY2 = cube_posXY2[:, 1] - 0.1
    # cube_pos_err2 = torch.sqrt((cube_posX2 - base_pos[:, 0]) * (cube_posX2 - base_pos[:, 0]) + (cube_posY2 - base_pos[:, 1]) * (cube_posY2 - base_pos[:, 1]))
    # cube_pos_err2 = 0.5 * torch.tanh(25*cube_pos_err2-3) + 0.495
    # cube_h3 = cube_states3[:, 2]
    # cube_posXY3 = cube_states3[:, 0:2]
    # cube_posX3 = cube_posXY3[:, 0] + 0.05
    # cube_posY3 = cube_posXY3[:, 1] - 0.1
    # cube_pos_err3 = torch.sqrt((cube_posX3 - base_pos[:, 0]) * (cube_posX3 - base_pos[:, 0]) + (cube_posY3 - base_pos[:, 1]) * (cube_posY3 - base_pos[:, 1]))
    # cube_pos_err3 = 0.5 * torch.tanh(25*cube_pos_err3-3) + 0.495
    #
    # cube_pos_err = (cube_pos_err + cube_pos_err2 + cube_pos_err3) / 3

    # movement of coxa
    m1 = torch.sqrt(dof_vel[:, 1]*dof_vel[:, 1])
    m2 = torch.sqrt(dof_vel[:, 4]*dof_vel[:, 4])
    m3 = torch.sqrt(dof_vel[:, 7] * dof_vel[:, 7])
    m4 = torch.sqrt(dof_vel[:, 10] * dof_vel[:, 10])
    m5 = torch.sqrt(dof_vel[:, 13] * dof_vel[:, 13])
    m6 = torch.sqrt(dof_vel[:, 16] * dof_vel[:, 16])
    total_movement = (m1 + m2 + m3 + m4 + m5 + m6)
    # punishment
    # p_m1 = torch.exp(-4.0 * m1) * 0.1
    # p_m2 = torch.exp(-4.0 * m2) * 0.1
    # p_m3 = torch.exp(-4.0 * m3) * 0.1
    # p_m4 = torch.exp(-4.0 * m4) * 0.1
    # p_m5 = torch.exp(-4.0 * m5) * 0.1
    # p_m6 = torch.exp(-4.0 * m6) * 0.1
    # p_m = p_m1 + p_m2 + p_m3 + p_m4 + p_m5 + p_m6

    #my total reward
    total_reward = lin_rw + ang_rw + rew_torque + rew_joint_acc + rew_action_rate + lin_pun + ang_pun - lin_x_err #- distance_err
    total_reward = torch.clip(total_reward, min=0., max=None)

    # reset agents
    reset = torch.norm(contact_forces[:, base_index, :], dim=1) > 10.
    #reset = reset | torch.any(torch.norm(contact_forces[:, knee_indices, :], dim=2) > 1., dim=1)
    reset = reset | (total_movement < 0.01)
    # reset = reset | (cube_h < 0.18)
    # reset = reset | (cube_h2 <= cube_h+0.05)
    # reset = reset | (cube_h3 <= cube_h+0.05)
    time_out = episode_lengths >= max_episode_length - 1  # no terminal reward for time-outs
    reset = reset | time_out

    #count += 1
    return total_reward.detach(), reset

@torch.jit.script
def compute_anymal_observations(root_states,
                                commands,
                                forces_sence,
                                dof_pos,
                                default_dof_pos,
                                dof_vel,
                                gravity_vec,
                                torques,
                                contact_forces,
                                actions,
                                #############
                                lin_vel_scale,
                                ang_vel_scale,
                                dof_pos_scale,
                                dof_vel_scale
                                ):

    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float) -> Tensor
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10]) * lin_vel_scale
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13]) * ang_vel_scale
    projected_gravity = quat_rotate(base_quat, gravity_vec)
    dof_pos_scaled = (dof_pos - default_dof_pos) * dof_pos_scale
    #print(base_lin_vel[0])
    #print(commands[0])
    commands_scaled = commands[:, 1:3]*torch.tensor([8, 1], requires_grad=False, device=commands.device)

    obs = torch.cat((base_lin_vel,
                     base_ang_vel,
                     projected_gravity,
                     commands_scaled,
                     forces_sence, # leo's stuff
                     dof_pos_scaled,
                     dof_vel*dof_vel_scale,
                     torques,
                     contact_forces[:, 3, :],
                     contact_forces[:, 6, :],
                     contact_forces[:, 9, :],
                     contact_forces[:, 12, :],
                     contact_forces[:, 15, :],
                     contact_forces[:, 18, :],
                     actions
                     ), dim=-1)
    # print('a1', contact_forces[9, 3, 2])
    # print(contact_forces[9, 6, 2])
    # print(contact_forces[9, 9, 2])
    # print(contact_forces[9, 12, 2])
    # print(contact_forces[9, 15, 2])
    # print(contact_forces[9, 18, 2])
    return obs

@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)