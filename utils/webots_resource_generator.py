from pathlib import Path
import yaml
import re

WEBOTS_WORLD_FILE_NAME = 'webots_world_file.wbt'


class WebotsResourceGenerator:
    """ Generates the proto and world files for the Webots environment """
    def __init__(self, cfg, paths):
        self.cfg = cfg
        self.paths = paths
    
    def generate_resources(self, map_name_list):
        print(f"Webots Resource Generator - Generating simulation resources based on data set configuration in {self.paths.data_sets.config}")
        
        data_set_config = self.load_data_set_config()
        map_name_list = self.load_map_name_list(data_set_config.list)
        
        # Generate proto files for each map
        for map_name in self.load_map_name_list(map_name_list):
            self.generate_proto(
                map_raster=map_raster, 
                output_file=Path(self.paths.data_sets.protos) / f"{map_name}.proto"
            )
        
        # Generate world file that includes all the map protos (required for importing the proto files)
        self.generate_world(
            input_world_file=Path(self.paths.resources.worlds) / WEBOTS_WORLD_FILE_NAME,
            output_world_file=Path(self.paths.data_sets.worlds)
        )
        
    def load_data_set_config(self):
        path_to_config = Path(self.paths.data_sets.config) / "config.yaml"
        with open(path_to_config) as f:
            return yaml.load(f, Loader=yaml.BaseLoader)
    
    def load_map_name_list(self, map_list_name):
        path_to_map_list = Path(self.paths.resources.map_name_lists) / f"{map_list_name}.yaml"   
        with open(path_to_map_list) as f:
            return yaml.load(f, Loader=yaml.BaseLoader)
        
    def generate_proto(self, map_raster: list, output_file):
        """ Saves the map as a proto file """
        with open(output_file, 'w') as f:
            self.convert_pgm_to_proto(self.cfg, map_raster, f, proto_name=output_file.stem)

    def generate_world(self, input_world_file, output_world_file):
        """Adds externproto import statements to the specified world file."""
        self.paths.data_sets.protos = Path(self.paths.data_sets.protos)
        file_name_list = [f.stem for f in self.paths.data_sets.protos.glob("*.proto")]
        
        # Read the world file
        with open(input_world_file, 'r') as wb_world_file:
            content = wb_world_file.read()
        
        # Add absolute file path for robot proto file
        content = content.replace("../protos", str(self.paths.resources.protos))    
        
        import_lines = "\n".join(
            f'IMPORTABLE EXTERNPROTO "{self.paths.data_sets.protos}/{file_name}.proto"' 
            for file_name in file_name_list
        )
        updated_content = re.sub(
            r'EXTERNPROTO "placeholder_start".*?EXTERNPROTO "placeholder_end"', 
            f'\n{import_lines}',
            content, 
            flags=re.DOTALL
        )
        
        # Write the updated world file
        with open(Path(output_world_file) / WEBOTS_WORLD_FILE_NAME, 'w') as write_file:
            write_file.write(updated_content)
    
    @staticmethod
    def replace_placeholders(content, substitutions):
        """ Replace placeholders in the content with values from the substitutions dictionary. """
        for key, value in substitutions.items():
            placeholder = f"{{{{{key}}}}}"
            if placeholder not in content:
                content = content.replace(placeholder, str(value))
                print(f"Replacing '{placeholder}' with '{value}' in the proto file.")
            content = content.replace(placeholder, str(value))
        return content

    @staticmethod
    def update_protos(cfg, path_to_proto):
        """ Update the proto files based on the provided configuration. """
        try:
            template_proto_file_name = cfg.template_name
            output_proto_file_name = cfg.output_name
            
            # Load the template proto file
            template_proto_file_path = find_file(
                filename=template_proto_file_name, 
                start_dir=path_to_proto
            )
            with open(template_proto_file_path, 'r') as file:
                template_proto_content = file.read()
        except FileNotFoundError:
            raise FileNotFoundError("Template proto file not found.")

        # Replace placeholders in the content with values from the config
        output_proto_content = WebotsAssetGenerator.replace_placeholders(
            template_proto_content, 
            cfg.substitutions
        )

        # Write the updated content to the output file
        output_proto_file_path = template_proto_file_path.replace(template_proto_file_name, output_proto_file_name)
        with open(output_proto_file_path, 'w') as file:
            file.write(output_proto_content)
            
    @staticmethod
    def add_vertex_points(points, index, i, x1, y1, x2, y2, image_height, origin, map_res):
        """append 4 3D points to the 'points' array,
        link to these points in the 'index' array,
        together the 4 points make up a 2D vertex, front and back"""
        points.append([-origin[0] + x1*map_res, -origin[1] + (image_height - y1)*map_res, origin[2]])
        points.append([-origin[0] + x2*map_res, -origin[1] + (image_height - y2)*map_res, origin[2]])
        points.append([-origin[0] + x2*map_res, -origin[1] + (image_height - y2)*map_res, origin[2] + 1])
        points.append([-origin[0] + x1*map_res, -origin[1] + (image_height - y1)*map_res, origin[2] + 1])

        # link to the points in the 'points' array, end with '-1'
        index.append([i, i+1, i+2, i+3, -1])  # front side of the vertex
        index.append([i+3, i+2, i+1, i, -1])  # back side of the vertex

        return i + 4
    
    def convert_pgm_to_proto(self, map_cfg, map_raster, output_file, proto_name):
        image_height = len(map_raster)
        origin = [0, image_height*map_cfg.resolution, 0]
        occupied_thresh = 255 * (1 - float(map_cfg.occupied_thresh))
        free_thresh = 255 * (1 - float(map_cfg.free_tresh))

        coords = []
        indices = []

        i = 0
        for r, row in enumerate(map_raster[1:-1], start=1):
            for c, pixel in enumerate(row[1:-1], start=1):
                # check if pixel == wall
                if pixel < occupied_thresh:
                    prev_i = i
                    points = []
                    index = []
                    # free space above pixel?
                    if map_raster[r-1][c] > free_thresh:
                        i = self.add_vertex_points(points, index, i, c, r, c+1, r, image_height, origin, map_cfg.resolution)
                    # free space below pixel?
                    if map_raster[r+1][c] > free_thresh:
                        i = self.add_vertex_points(points, index, i, c, r+1, c+1, r+1, image_height, origin, map_cfg.resolution)
                    # free space left pixel?
                    if map_raster[r][c-1] > free_thresh:
                        i = self.add_vertex_points(points, index, i, c, r, c, r+1, image_height, origin, map_cfg.resolution)
                    # free space right pixel?
                    if map_raster[r][c+1] > free_thresh:
                        i = self.add_vertex_points(points, index, i, c+1, r, c+1, r+1, image_height, origin, map_cfg.resolution)
                    # new indexFace added?
                    if i > prev_i:
                        coords.append(points)
                        indices.append(index)

        pf = open(output_file.name, 'w+')

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