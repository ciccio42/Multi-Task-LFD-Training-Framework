from collections import OrderedDict
import numpy as np

from robosuite.utils.transform_utils import convert_quat, quat2mat, mat2quat
from robosuite.utils.mjcf_utils import CustomMaterial

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.objects import (
    MilkVisualObject,
    BreadVisualObject,
    CerealVisualObject,
    CanVisualObject,
)
from multi_task_robosuite_env.arena import TableArena, BinsArena
from multi_task_robosuite_env.objects.custom_xml_objects import *
from multi_task_robosuite_env.sampler import BoundarySampler
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from multi_task_robosuite_env.objects.meta_xml_objects import Bin
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, find_elements
import robosuite.utils.transform_utils as T

OFFSET = 0.0


class PickPlace(SingleArmEnv):
    """
    This class corresponds to the stacking task for a single robot arm.
    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!
        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.
        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param
        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param
        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:
            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"
            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param
            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.
        table_full_size (3-tuple): x, y, and z dimensions of the table.
        table_friction (3-tuple): the three mujoco friction parameters for
            the table.
        use_camera_obs (bool): if True, every observation includes rendered image(s)
        use_object_obs (bool): if True, include object (cube) information in
            the observation.
        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized
        reward_shaping (bool): if True, use dense rewards.
        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.
        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.
        has_offscreen_renderer (bool): True if using off-screen rendering
        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse
        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.
        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.
        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).
        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.
        horizon (int): Every episode lasts for exactly @horizon timesteps.
        ignore_done (bool): True if never terminating the environment (ignore @horizon).
        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables
        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.
            :Note: At least one camera must be specified if @use_camera_obs is True.
            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).
        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.
        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.
        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.
    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
            self,
            robots,
            env_configuration="default",
            controller_configs=None,
            mount_types="default",
            gripper_types="default",
            robot_offset=None,
            initialization_noise="default",
            table_full_size=(0.8, 0.8, 0.05),
            table_friction=(1., 5e-3, 1e-4),
            table_offset=(0, 0, 0.82),
            use_camera_obs=True,
            use_object_obs=True,
            reward_scale=1.0,
            reward_shaping=False,
            placement_initializer=None,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="frontview",
            render_collision_mesh=False,
            render_visual_mesh=True,
            render_gpu_device_id=-1,
            control_freq=20,
            horizon=1000,
            ignore_done=False,
            hard_reset=True,
            camera_names="agentview",
            camera_heights=256,
            camera_widths=256,
            camera_depths=False,
            camera_poses=None,
            camera_attribs=None,
            camera_gripper=None,
            task_id=0,
            object_type=None,
            y_ranges=[[0.16, 0.19], [0.05, 0.09],
                      [-0.08, -0.03], [-0.19, -0.15]],
            env_conf=None,
            arena="Table"
    ):
        print(f"Render GPU id {render_gpu_device_id}")
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array(table_offset)
        self.robot_offset = robot_offset
        self.object_id = int(task_id//4)
        self.bin_id = task_id % 4
        self.bin_size = np.array((0.16, 0.16))
        self.object_set = env_conf['object_set']
        print(f"Object set {self.object_set}")

        if self.object_set == 1:
            self.object_to_id = {"milk": 0, "bread": 1, "cereal": 2, "can": 3}
            self.obj_names = ["Milk", "Bread", "Cereal", "Can"]
            self._obj_dim = {'milk': [0.05, 0.12, 0.045],
                             'bread': [0.045, 0.055, 0.045],
                             'cereal': [0.045, 0.055, 0.045],
                             'can': [0.045, 0.065, 0.045]}
        elif self.object_set == 2:
            self.object_to_id = {"greenbox": 0,
                                 "yellowbox": 1,
                                 "bluebox": 2,
                                 "redbox": 3}
            self.obj_names = ["greenbox",
                              "yellowbox",
                              "bluebox",
                              "redbox"]
            self._obj_dim = {'greenbox': [0.05, 0.055, 0.045],
                             'yellowbox': [0.05, 0.055, 0.045],
                             'bluebox': [0.05, 0.055, 0.045],
                             'redbox': [0.05, 0.055, 0.045],
                             'bin': [0.64, 0.07, 0.16]}
        else:
            self.init_object_dicts()

        if object_type is not None:
            assert (
                object_type in self.object_to_id.keys()
            ), "invalid @object_type argument - choose one of {}".format(
                list(self.object_to_id.keys())
            )
            self.object_id = self.object_to_id[
                object_type
            ]  # use for convenient indexing

        self.y_ranges = y_ranges
        self.x_ranges = env_conf['x_ranges']

        self.bin_position_x = env_conf['bin_position_x']
        self.bin_position_y = env_conf['bin_position_y']

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # camera poses and attributes
        self.camera_names = camera_names
        self.camera_poses = camera_poses
        self.camera_attribs = camera_attribs
        self.camera_gripper = camera_gripper
        self.camera_height = camera_heights
        self.camera_width = camera_widths

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types=mount_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=env_conf['ignore_done'],
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def init_object_dicts(self, object_id=0):
        box_names = ["greenbox",
                     "yellowbox",
                     "bluebox",
                     "redbox"]
        box_dims = {'greenbox': [0.05, 0.055, 0.045],
                    'yellowbox': [0.05, 0.055, 0.045],
                    'bluebox': [0.05, 0.055, 0.045],
                    'redbox': [0.05, 0.055, 0.045],
                    'bin': [0.64, 0.07, 0.16]}

        nut_names = ['nut_gray',
                     'nut_ambra',
                     'nut_blue']
        nut_dims = {'nut_gray': [0.16, 0.02, 0.16],
                    'nut_ambra': [0.16, 0.02, 0.16],
                    'nut_blue': [0.16, 0.02, 0.16],
                    'peg1': [0.06, 0.06, 0.11],
                    'peg2': [0.06, 0.06, 0.11],
                    'peg3': [0.06, 0.06, 0.11]}

        self.object_to_id = dict()
        self.obj_names = list()
        self._obj_dim = dict()

        for i in range(4):
            # put the target nut insted of target box
            if i == object_id and i < 3:
                self.object_to_id[nut_names[i]] = i
                self.obj_names.append(nut_names[i])
                self._obj_dim[nut_names[i]] = nut_dims[nut_names[i]]
            elif i != object_id and i < 3:
                # put random nut insted of box
                sub_box = np.random.choice([True, False], 1, p=[0.5, 0.5])
                if sub_box:
                    self.object_to_id[nut_names[i]] = i
                    self.obj_names.append(nut_names[i])
                    self._obj_dim[nut_names[i]] = nut_dims[nut_names[i]]
                else:
                    self.object_to_id[box_names[i]] = i
                    self.obj_names.append(box_names[i])
                    self._obj_dim[box_names[i]] = box_dims[box_names[i]]
            elif i == object_id and i == 3:
                self.object_to_id[box_names[i]] = i
                self.obj_names.append(box_names[i])
                self._obj_dim[box_names[i]] = box_dims[box_names[i]]
            elif i != object_id and i == 3:
                # select random box
                box_names_sample = [box_names[object_id], box_names[i]]
                box_name = np.random.choice(box_names_sample, 1)[0]
                self.object_to_id[box_name] = i
                self.obj_names.append(box_name)
                self._obj_dim[box_name] = box_dims[box_name]

        self._obj_dim['bin'] = [0.64, 0.07, 0.16]

    def reward(self, action):
        """
        Reward function for the task.
        Sparse un-normalized reward:
            - a discrete reward of 2.0 is provided if the red block is stacked on the green block
        Un-normalized components if using reward shaping:
            - Reaching: in [0, 0.25], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube
            - Aligning: in [0, 0.5], encourages aligning one cube over the other
            - Stacking: in {0, 2}, non-zero if cube is stacked on other cube
        The reward is max over the following:
            - Reaching + Grasping
            - Lifting + Aligning
            - Stacking
        The sparse reward only consists of the stacking component.
        Note that the final reward is normalized and scaled by
        reward_scale / 2.0 as well so that the max score is equal to reward_scale
        Args:
            action (np array): [NOT USED]
        Returns:
            float: reward value
        """
        reward = float(self._check_success())

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        if self.robot_offset is None:
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](
                self.table_full_size[0])
        else:
            xpos = self.robot_offset
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # add desired cameras
        if self.camera_poses is not None:
            for camera_name in self.camera_names:
                if camera_name != "robot0_eye_in_hand":
                    mujoco_arena.set_camera(camera_name=camera_name,
                                            pos=self.camera_poses[camera_name][0],
                                            quat=self.camera_poses[camera_name][1],
                                            camera_attribs=self.camera_attribs)

        # modify robot0_eye_in_hand
        if self.robots[0].robot_model.default_gripper == "Robotiq85Gripper":
            self.robots[0].robot_model.set_camera(camera_name="eye_in_hand",
                                                  pos=self.camera_gripper["Robotiq85Gripper"]["pose"][0],
                                                  quat=self.camera_gripper["Robotiq85Gripper"]["pose"][1],
                                                  root=self.camera_gripper["Robotiq85Gripper"]["root"],
                                                  camera_attribs=self.camera_attribs)

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        self.bin = [Bin(name='bin')]
        # Create placement initializer

        self.objects = []
        self.visual_objects = []
        self.object_bin = []

        if self.object_set == 1:
            for vis_obj_cls, obj_name in zip(
                    (MilkVisualObject, BreadVisualObject,
                     CerealVisualObject, CanVisualObject),
                    self.obj_names,
            ):
                vis_name = "Visual" + obj_name
                vis_obj = vis_obj_cls(name=vis_name)
                self.visual_objects.append(vis_obj)

        if self.object_set == 1:
            object_seq = (MilkObject, BreadObject, CerealObject, CanObject)
        elif self.object_set == 2:
            object_seq = (BoxObject, BoxObject, BoxObject, BoxObject)
        elif self.object_set == 3:
            object_seq = list()
            for obj_name in self.obj_names:
                if "nut" in obj_name:
                    object_seq.append(NamedRoundNutObject)
                else:
                    object_seq.append(BoxObject)

        for obj_cls, obj_name in zip(
                object_seq,
                self.obj_names,
        ):
            obj = obj_cls(name=obj_name)
            self.objects.append(obj)
            self.object_bin.append(obj)

        self.object_bin.append(self.bin[0])

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.visual_objects + self.objects + self.bin,
        )

        self._get_placement_initializer()

        compiler = self.model.root.find('compiler')
        compiler.set('inertiafromgeom', 'auto')
        if compiler.attrib['inertiagrouprange'] == "0 0":
            compiler.attrib.pop('inertiagrouprange')

    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds.
        """
        self.placement_initializer = SequentialCompositeSampler(
            name="ObjectSampler")
        arr = np.arange(4)
        np.random.shuffle(arr)

        for i in range(4):
            if "nut" in self.obj_names[i]:
                rotation = [3.14, 3.14 + 1e-4]
            else:
                rotation = [0, 0 + 1e-4]

            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name='obj'+str(i)+"Sampler",
                    mujoco_objects=self.objects[i],
                    x_range=self.x_ranges[0],
                    y_range=self.y_ranges[arr[i]],
                    rotation=rotation,
                    rotation_axis='z',
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.0001,
                )
            )

        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="BinSampler",
                mujoco_objects=self.bin,
                x_range=[self.bin_position_x[0],
                         self.bin_position_x[1] + 1e-4],
                y_range=[self.bin_position_y[0],
                         self.bin_position_y[1] + 1e-4],
                rotation=[0, 0 + 1e-4],
                rotation_axis='z',
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=False,
                reference_pos=self.table_offset,
                z_offset=0.0,
            )
        )

    def _create_bb(self, di):
        """
            Create bb around each object in the scene for each camera of interest
        """
        import logging
        logging.basicConfig(
            format='%(levelname)s:%(message)s', level=logging.INFO)
        logger = logging.getLogger("BB-Creator")

        obj_bb = OrderedDict()
        # 1. For each camera of interest get the pose
        for camera_name in self.camera_names:
            if "in_hand" not in camera_name:
                obj_bb[camera_name] = OrderedDict()
                r_camera_world = quat2mat(convert_quat(
                    np.array(self.camera_poses[camera_name][1]), to="xyzw")).T
                p_camera_world = - \
                    r_camera_world @  np.reshape(np.expand_dims(
                        np.array(self.camera_poses[camera_name][0]), axis=0), (3, 1))
                # Transformation matrix
                T_camera_world = np.concatenate(
                    (r_camera_world, p_camera_world), axis=1)
                T_camera_world = np.concatenate(
                    (T_camera_world, np.array([[0, 0, 0, 1]])), axis=0)

                # 2. For each object compute bb
                for i, obj in enumerate(self.object_bin):
                    obj_name = obj.name
                    obj_bb[camera_name][obj_name] = dict()

                    # convert obj pos in camera coordinate
                    obj_pos = np.array(
                        self.sim.data.body_xpos[self.obj_body_id[obj_name]])
                    if obj_name == 'bin':
                        obj_pos[2] = obj_pos[2] + 0.08
                    obj_quat = T.convert_quat(
                        self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw"
                    )

                    # 2. Create transformation matrix
                    T_camera_world = np.concatenate(
                        (r_camera_world, p_camera_world), axis=1)
                    T_camera_world = np.concatenate(
                        (T_camera_world, np.array([[0, 0, 0, 1]])), axis=0)
                    # logger.debug(T_camera_world)
                    p_world_object = np.expand_dims(
                        np.insert(obj_pos, 3, 1), 0).T
                    p_camera_object = T_camera_world @ p_world_object
                    logger.debug(
                        f"\nP_world_object:\n{p_world_object} - \nP_camera_object:\n {p_camera_object}")

                    # 3. Cnversion into pixel coordinates of object center
                    f = 0.5 * self.camera_height / \
                        np.tan(int(self.camera_attribs['fovy']) * np.pi / 360)

                    p_x_center = int(
                        (p_camera_object[0][0] / - p_camera_object[2][0]) * f + self.camera_width / 2)

                    p_y_center = int(
                        (- p_camera_object[1][0] / - p_camera_object[2][0]) * f + self.camera_height / 2)
                    logger.debug(
                        f"\nImage coordinate: px {p_x_center}, py {p_y_center}")

                    p_x_corner_list = []
                    p_y_corner_list = []
                    # 3.1 create a box around the object
                    for i in range(8):
                        if i == 0:  # upper-left front corner
                            p_world_object_corner = p_world_object + \
                                np.array(
                                    [[self._obj_dim[obj_name][2]/2],
                                        [-self._obj_dim[obj_name][0]/2-OFFSET],
                                        [self._obj_dim[obj_name][1]/2+OFFSET],
                                        [0]])
                        elif i == 1:  # upper-right front corner
                            p_world_object_corner = p_world_object + \
                                np.array(
                                    [[self._obj_dim[obj_name][2]/2],
                                        [self._obj_dim[obj_name][0]/2+OFFSET],
                                        [self._obj_dim[obj_name][1]/2+OFFSET],
                                        [0]])
                        elif i == 2:  # bottom-left front corner
                            p_world_object_corner = p_world_object + \
                                np.array(
                                    [[self._obj_dim[obj_name][2]/2],
                                        [-self._obj_dim[obj_name][0]/2-OFFSET],
                                        [-self._obj_dim[obj_name][1]/2-OFFSET],
                                        [0]])
                        elif i == 3:  # bottom-right front corner
                            p_world_object_corner = p_world_object + \
                                np.array(
                                    [[self._obj_dim[obj_name][2]/2],
                                        [self._obj_dim[obj_name][0]/2+OFFSET],
                                        [-self._obj_dim[obj_name][1]/2-OFFSET],
                                        [0]])
                        elif i == 4:  # upper-left back corner
                            p_world_object_corner = p_world_object + \
                                np.array(
                                    [[-self._obj_dim[obj_name][2]/2],
                                        [-self._obj_dim[obj_name][0]/2-OFFSET],
                                        [self._obj_dim[obj_name][1]/2+OFFSET],
                                        [0]])
                        elif i == 5:  # upper-right back corner
                            p_world_object_corner = p_world_object + \
                                np.array(
                                    [[-self._obj_dim[obj_name][2]/2],
                                        [self._obj_dim[obj_name][0]/2+OFFSET],
                                        [self._obj_dim[obj_name][1]/2+OFFSET],
                                        [0]])
                        elif i == 6:  # bottom-left back corner
                            p_world_object_corner = p_world_object + \
                                np.array(
                                    [[-self._obj_dim[obj_name][2]/2],
                                        [-self._obj_dim[obj_name][0]/2-OFFSET],
                                        [-self._obj_dim[obj_name][1]/2-OFFSET],
                                        [0]])
                        elif i == 7:  # bottom-right back corner
                            p_world_object_corner = p_world_object + \
                                np.array(
                                    [[-self._obj_dim[obj_name][2]/2],
                                        [self._obj_dim[obj_name][0]/2+OFFSET],
                                        [-self._obj_dim[obj_name][1]/2-OFFSET],
                                        [0]])

                        p_camera_object_corner = T_camera_world @ p_world_object_corner
                        logger.debug(
                            f"\nP_world_object_upper_left:\n{p_world_object_corner} -   \nP_camera_object_upper_left:\n {p_camera_object_corner}")

                        # 3.1 Upper-left corner and bottom right corner in pixel coordinate
                        p_x_corner = int(
                            (p_camera_object_corner[0][0] / - p_camera_object_corner[2][0]) * f + self.camera_width / 2)

                        p_y_corner = int(
                            (- p_camera_object_corner[1][0] / - p_camera_object_corner[2][0]) * f + self.camera_height / 2)
                        logger.debug(
                            f"\nImage coordinate upper_left corner: px {p_x_corner}, py {p_y_corner}")

                        p_x_corner_list.append(p_x_corner)
                        p_y_corner_list.append(p_y_corner)

                    x_min = min(p_x_corner_list)
                    y_min = min(p_y_corner_list)
                    x_max = max(p_x_corner_list)
                    y_max = max(p_y_corner_list)
                    # save bb
                    obj_bb[camera_name][obj_name]['center'] = [
                        p_x_center, p_y_center]
                    obj_bb[camera_name][obj_name]['upper_left_corner'] = [
                        x_max, y_max]
                    obj_bb[camera_name][obj_name]['bottom_right_corner'] = [
                        x_min, y_min]

        return obj_bb

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:
            robot-state: contains robot-centric information.
            object-state: requires @self.use_object_obs to be True.
                contains object-centric information.
            image: requires @self.use_camera_obs to be True.
                contains a rendered frame from the simulation.
            depth: requires @self.use_camera_obs and @self.camera_depth to be True.
                contains a rendered depth map from the simulation
        """
        di = super()._get_observation()

        di['target-box-id'] = self.bin_id
        di['target-object'] = self.object_id

        # add observation for all objects
        pr = self.robots[0].robot_model.naming_prefix

        # remember the keys to collect into object info
        object_state_keys = []

        # for conversion to relative gripper frame
        gripper_pose = T.pose2mat(
            (di[pr + "eef_pos"], di[pr + "eef_quat"]))
        world_pose_in_gripper = T.pose_inv(gripper_pose)

        for i, obj in enumerate(self.objects):
            obj_str = obj.name
            if "nut" in obj_str:
                obj_pos = self.sim.data.site_xpos[self.sim.model.site_name2id(
                    f'{obj_str}_handle_site')]
                obj_quat = T.mat2quat(
                    np.reshape(self.sim.data.site_xmat[self.sim.model.site_name2id(
                        f'{obj_str}_handle_site')], (3, 3)))
            else:
                obj_pos = np.array(
                    self.sim.data.body_xpos[self.obj_body_id[obj_str]])
                obj_quat = T.convert_quat(
                    self.sim.data.body_xquat[self.obj_body_id[obj_str]], to="xyzw"
                )
            di["{}_pos".format(obj_str)] = obj_pos
            di["{}_quat".format(obj_str)] = obj_quat

            # get relative pose of object in gripper frame
            object_pose = T.pose2mat((obj_pos, obj_quat))
            rel_pose = T.pose_in_A_to_pose_in_B(
                object_pose, world_pose_in_gripper)
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            di["{}_to_{}eef_pos".format(obj_str, pr)] = rel_pos
            di["{}_to_{}eef_quat".format(obj_str, pr)] = rel_quat

            object_state_keys.append("{}_pos".format(obj_str))
            object_state_keys.append("{}_quat".format(obj_str))
            object_state_keys.append("{}_to_{}eef_pos".format(obj_str, pr))
            object_state_keys.append(
                "{}_to_{}eef_quat".format(obj_str, pr))

        for i, bin_name in enumerate(['bin_box_1', 'bin_box_2', 'bin_box_3', 'bin_box_4']):

            bin_bodyid = self.sim.model.body_name2id(bin_name)

            bin_pos = np.array(self.sim.data.body_xpos[bin_bodyid])

            di["{}_pos".format(bin_name)] = bin_pos

        di["object-state"] = np.concatenate([di[k]
                                            for k in object_state_keys])

        di['obj_bb'] = self._create_bb(di)

        return di

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()
        self.obj_body_id = {}
        self.obj_geom_id = {}

        # object-specific ids
        for obj in (self.visual_objects + self.objects + self.bin):
            self.obj_body_id[obj.name] = self.sim.model.body_name2id(
                obj.root_body)
            self.obj_geom_id[obj.name] = [
                self.sim.model.geom_name2id(g) for g in obj.contact_geoms]

        # keep track of which objects are in their corresponding bins
        self.objects_in_bins = np.zeros(len(self.objects))
        names = ['bin_box_1', 'bin_box_2', 'bin_box_3', 'bin_box_4']
        self.bin_bodyid = self.sim.model.body_name2id(names[self.bin_id])

    def not_in_bin(self, obj_pos):
        self.bin_pos = np.array(self.sim.data.body_xpos[self.bin_bodyid])
        bin_x_low = self.bin_pos[0]
        bin_y_low = self.bin_pos[1]
        bin_x_low -= self.bin_size[0] / 2
        bin_y_low -= self.bin_size[1] / 2

        bin_x_high = bin_x_low + self.bin_size[0]
        bin_y_high = bin_y_low + self.bin_size[1]
        # print(self.bin_pos, obj_pos)
        res = True
        if (
                bin_x_low < obj_pos[0] < bin_x_high
                and bin_y_low < obj_pos[1] < bin_y_high
                and self.bin_pos[2] < obj_pos[2] < self.bin_pos[2] + 0.1
        ):
            res = False
        return res

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[-1], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def _check_success(self):
        obj_str = self.objects[self.object_id].name
        obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
        return not self.not_in_bin(obj_pos)

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.
        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper, target=self.drawers[self.drawer_id])


class UR5ePickPlace(PickPlace):
    """
    Easier version of task - place one object into its bin.
    A new object is sampled on every reset.
    """

    def __init__(self, task_id=None, **kwargs):
        if task_id is None:
            task_id = np.random.randint(0, 8)
        super().__init__(robots=['UR5e'], task_id=task_id, **kwargs)


class PandaPickPlace(PickPlace):
    """
    Easier version of task - place one object into its bin.
    A new object is sampled on every reset.
    """

    def __init__(self, task_id=None, **kwargs):
        if task_id is None:
            task_id = np.random.randint(0, 8)
        super().__init__(robots=['Panda'], task_id=task_id, **kwargs)


class SawyerPickPlace(PickPlace):
    """
    Easier version of task - place one object into its bin.
    A new object is sampled on every reset.
    """

    def __init__(self, task_id=None, **kwargs):
        if task_id is None:
            task_id = np.random.randint(0, 8)
        super().__init__(robots=['Sawyer'], task_id=task_id, **kwargs)


if __name__ == '__main__':
    from robosuite.environments.manipulation.pick_place import PickPlace
    import robosuite
    from robosuite.controllers import load_controller_config
    import debugpy
    import yaml
    import cv2

    debugpy.listen(('0.0.0.0', 5678))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()

    controller = load_controller_config(default_controller="IK_POSE")

    # load env conf
    with open('/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/config/PickPlaceDistractor.yaml', 'r') as file:
        env_conf = yaml.safe_load(file)

    env_conf['object_set'] = 3
    env = UR5ePickPlace(task_id=0,
                        has_renderer=False,
                        controller_configs=controller,
                        has_offscreen_renderer=True,
                        reward_shaping=False,
                        use_camera_obs=True,
                        camera_heights=env_conf['camera_heights'],
                        camera_widths=env_conf['camera_widths'],
                        render_camera='frontview',
                        camera_depths=False,
                        camera_names=env_conf['camera_names'],
                        camera_poses=env_conf['camera_poses'],
                        camera_attribs=env_conf['camera_attribs'],
                        camera_gripper=env_conf['camera_gripper'],
                        env_conf=env_conf,)
    env.reset()
    for i in range(1000):
        if i % 200 == 0:
            env.reset()
        low, high = env.action_spec
        action = np.random.uniform(low=low, high=high)
        obs = env.step(action)
        img = obs[0]['camera_front_image']
        env.render()
