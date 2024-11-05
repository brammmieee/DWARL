from pathlib import Path
import yaml
import re
import numpy as np
from omegaconf import OmegaConf
import shutil

class WebotsResourceGenerator:
    """ Generates the proto and world files for the Webots environment """
    def __init__(self, cfg, paths):
        self.cfg = cfg
        self.paths = paths
    
    def erase_old_data(self):
        sim_resource_folder = Path(self.paths.sim_resources.root)
        if sim_resource_folder.exists() and sim_resource_folder.is_dir():
            shutil.rmtree(sim_resource_folder)
            
    def create_folder_structure(self):
        Path(self.paths.sim_resources.root).mkdir(parents=True, exist_ok=True)
        for sim_resource_path in self.paths.sim_resources.values():
            Path(sim_resource_path).mkdir(parents=True, exist_ok=True)
    
    def generate_resources(self):
        # Generate the folder structure
        self.create_folder_structure()
        
        # Generate proto files for each map
        path_to_maps = Path(self.paths.data_sets.maps)
        for path_to_map in path_to_maps.glob("*.npy"):
            map_box_array = np.load(path_to_map)
            map_name = path_to_map.stem
            path_to_proto = Path(self.paths.sim_resources.protos) / f"{map_name}.proto"
            self.convert_box_array_to_proto(
                box_array=map_box_array,
                output_file=path_to_proto,
                proto_name=map_name
            )
        
        # Generate world file that includes all the map protos (required for importing the proto files)
        self.generate_world(
            input_world_file=Path(self.paths.resources.worlds) / self.cfg.world_file_name,
            output_world_file=Path(self.paths.sim_resources.worlds),
            path_to_protos=Path(self.paths.sim_resources.protos)
        )
    
    def generate_world(self, input_world_file, output_world_file, path_to_protos):
        """Adds externproto import statements to the specified world file."""        
        # Read the world file
        with open(input_world_file, 'r') as wb_world_file:
            content = wb_world_file.read()
        
        # Add absolute file path for robot proto file
        content = content.replace("../protos", str(self.paths.resources.protos))    
        import_lines = "\n".join(
            f'IMPORTABLE EXTERNPROTO "{self.paths.sim_resources.protos}/{file_name}.proto"' 
            for file_name in [f.stem for f in path_to_protos.glob("*.proto")]
        )
        updated_content = re.sub(
            r'EXTERNPROTO "placeholder_start".*?EXTERNPROTO "placeholder_end"', 
            f'\n{import_lines}',
            content, 
            flags=re.DOTALL
        )
        
        # Write the updated world file
        with open(Path(output_world_file) / self.cfg.world_file_name, 'w') as write_file:
            write_file.write(updated_content)
            
    @staticmethod
    def convert_box_array_to_proto(box_array, output_file, proto_name):
        coords = []
        indices = []
        i = 0
        for box in box_array:
            points = []
            index = []
            for vertex in box:
                points.append(vertex.tolist() + [0])  # Add z-coordinate as 0
                points.append(vertex.tolist() + [1])  # Add z-coordinate as 1
            
            # Create indices for each face of the box
            for j in range(4):  # 4 sides of the box
                front_face = [i+(j*2), i+((j*2+2)%8), i+((j*2+3)%8), i+(j*2+1), -1]
                back_face = [i+(j*2+1), i+((j*2+3)%8), i+((j*2+2)%8), i+(j*2), -1]
                index.append(front_face)
                index.append(back_face)
            
            coords.append(points)
            indices.append(index)
            i += 8  # Increment by 8 because we added 8 points (4 vertices * 2 for top and bottom)
            
        with open(output_file, 'w') as f:
            pf = open(f.name, 'w+')

            # Write proto-file header
            pf.write('#VRML_SIM R2019a utf8\n')
            pf.write('# license: Apache License 2.0\n')
            pf.write('# license url: http://www.apache.org/licenses/LICENSE-2.0\n')
            pf.write('\n')

            # Add appearance EXTERNPROTO
            pf.write('EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2022b/projects/robots/kuka/youbot/protos/BlackMetalAppearance.proto"')
            pf.write('\n\n')

            # Define PROTO with name 'map' and params translation and rotation
            pf.write('PROTO ' + proto_name + ' [\n')
            pf.write('  field  SFVec3f     translation     0 0 0\n')
            pf.write('  field  SFRotation  rotation        0 0 0 0\n')
            pf.write(']\n')
            pf.write('{\n')

            # Transform map based on params
            pf.write('  Transform {\n')
            pf.write('      translation IS translation\n')
            pf.write('      rotation    IS rotation\n')
            pf.write('      children [\n')

            # Open Shape with BlackMetalAppearance
            pf.write('          Shape {\n')
            pf.write('              appearance BlackMetalAppearance {\n')
            pf.write('              }\n')
            pf.write('              geometry IndexedFaceSet {\n')
            pf.write('                  coord Coordinate {\n')
            pf.write('                      point [\n')

            # Write all coordinates
            pf.write('                          ')
            for p in coords:
                for i in p:
                    for j in i:
                        pf.write(' ' + str(j))
                    pf.write(',')
            pf.write('\n                      ]\n')
            pf.write('                  }\n')
            pf.write('                  coordIndex [\n')

            # Write all indices that point to the coordinates
            pf.write('                      ')
            for p in indices:
                for i in p:
                    for j in i:
                        pf.write(' ' + str(j))
            pf.write('\n                ]\n')
            pf.write('                  creaseAngle 0\n')
            pf.write('              }\n')
            pf.write('          }\n')
            pf.write('      ]\n')
            pf.write('  }\n')
            pf.write('}\n')

            pf.close()