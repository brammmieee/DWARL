from controller import Supervisor
from pathlib import Path
from subprocess import Popen, PIPE
import numpy as np
import os
from utils.webots_resource_generator import WebotsResourceGenerator

STATIC_ROBOT_Z_POS = 0.05
WEBOTS_WORLD_FILE_NAME = 'webots_world_file.wbt'


class WebotsEnv(Supervisor):
    def __init__(self, cfg, paths):
        self.webots_generator = WebotsResourceGenerator(cfg.map, paths)
        self.webots_generator.generate_resources()

        # Open the world
        world_file = Path(paths.data_sets.worlds) / WEBOTS_WORLD_FILE_NAME
        self.open_world(cfg.mode, world_file)
        
        # Connect to the supervisor robot
        super().__init__()

        self.basic_timestep = int(self.getBasicTimeStep())
        self.timestep = 2*self.basic_timestep #NOTE: basic timestep set 0.5*timestep for lidar update
        
        # Static node references
        self.robot_node = self.getFromDef('ROBOT')
        self.root_node = self.getRoot() # root node (the nodes seen in the Webots scene tree editor window are children of the root node)
        self.robot_translation_field = self.robot_node.getField('translation')
        self.robot_rotation_field = self.robot_node.getField('rotation')
        self.root_children_field = self.root_node.getField('children') # used for inserting map node

        # Lidar sensor and keyboard
        self.lidar_node = self.getDevice('lidar')
        self.lidar_node.enable(int(self.getBasicTimeStep()))
    
    def reset(self):
        self.simulationReset()
        super().step(self.basic_timestep) # super prevents confusion with self.step() defined below

    def reset_map(self, proto_name):
        # Loading and translating map into position
        self.root_children_field.importMFNodeFromString(position=-1, nodeString='DEF MAP ' + proto_name + '{}')
        super().step(self.basic_timestep)

    def reset_robot(self, init_pose):
        # Positioning the robot at init_pos
        print(f"Setting robot position to {init_pose}")
        
        self.robot_translation_field.setSFVec3f([init_pose[0], init_pose[1], STATIC_ROBOT_Z_POS])
        self.robot_rotation_field.setSFRotation([0.0, 0.0, 1.0, init_pose[3]])
        super().step(2*self.basic_timestep) #NOTE: 2 timesteps needed in order to succesfully set the init position
        # NOTE: could the issue be caused by the lidar frequency?
        # NOTE: could the issue be caused by basic timestep defined in the world file?
        # NOTE: could the issue be caused by the FPS of the simulation?
    
    def step(self, new_position, new_orientation):
        self.robot_translation_field.setSFVec3f([new_position[0], new_position[1], new_position[2]])
        self.robot_rotation_field.setSFRotation([0.0, 0.0, 1.0, new_orientation])
        super().step(self.basic_timestep)
        super().step(self.basic_timestep) #NOTE: only after this timestep will the lidar data of the previous step be available

    def close_webots(self):
        self.simulationQuit(0)
        self.killall()

    @property
    def robot_position(self):
        return np.array(self.robot_node.getPosition())

    @property
    def robot_orientation(self):
        return np.array(self.robot_node.getOrientation())
    
    @property
    def lidar_resolution(self):
        return self.lidar_node.getHorizontalResolution()
    
    @staticmethod
    def killall():
        command = "ps aux | grep webots | grep -v grep | awk '{print $2}' | xargs -r kill"
        os.system(command)

    @staticmethod
    def open_world(mode, world_file):   
        # Create Webots command with specified mode and world file
        cmd = ['webots','--extern-urls', '--no-rendering', f'--mode={mode}', world_file]

        # Open Webots
        wb_process = Popen(cmd, stdout=PIPE)

        # Set the environment variable for the controller to connect to the supervisor
        output = wb_process.stdout.readline().decode("utf-8")
        ipc_prefix = 'ipc://'
        start_index = output.find(ipc_prefix)
        port_nr = output[start_index + len(ipc_prefix):].split('/')[0]
        os.environ["WEBOTS_CONTROLLER_URL"] = ipc_prefix + str(port_nr)