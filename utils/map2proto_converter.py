import os
import yaml
from ament_index_python.packages import get_package_share_directory, get_package_prefix

class Map2ProtoConverter:
    def __init__(self):
        self.package_dir = os.path.dirname(os.getcwd())

        self.map_dir = os.path.join(self.package_dir, 'maps')
        self.map_proto_dir = os.path.join(self.package_dir, 'protos', 'converted_maps')
        self.world_file_dir = os.path.join(self.package_dir, 'worlds')

    def read_pgm(self, pgmf):
        """Return a raster of integers from a PGM as a list of lists."""
        # Read header information 
        line_number = 0 
        while line_number < 2:
            line = pgmf.readline()
            if line_number == 0:  # Magic num info
                P_type = line.strip()
            if P_type != b'P2' and P_type != b'P5':
                pgmf.close()
                print('Not a valid PGM file')
                exit()
            if line_number == 1:  # Width, Height and Depth
                [width, height, depth] = (line.strip()).split()
                width = int(width)
                height = int(height)
                depth = int(depth)
            line_number += 1

        raster = []
        for _ in range(height):
            row = []
            for _ in range(width):
                row.append(ord(pgmf.read(1)))
            raster.append(row)
        return raster

    def add_vertex_points(self, points, index, i, x1, y1, x2, y2, image_height, origin, map_res):
        """append 4 3D points to the 'points' array,
        link to these points in the 'index' array,
        together the 4 points make up a 2D vertex, front and back"""
        points.append([-origin[0] + x1*map_res, 0, (y1 - image_height)*map_res + origin[1]])
        points.append([-origin[0] + x2*map_res, 0, (y2 - image_height)*map_res + origin[1]])
        points.append([-origin[0] + x2*map_res, 1, (y2 - image_height)*map_res + origin[1]])
        points.append([-origin[0] + x1*map_res, 1, (y1 - image_height)*map_res + origin[1]])

        # link to the points in the 'points' array, end with '-1'
        index.append([i, i+1, i+2, i+3, -1])  # front side of the vertex
        index.append([i+3, i+2, i+1, i, -1])  # back side of the vertex

        return i + 4

    def convert_pgm_to_proto(self, file_name, path_to_pgm, path_to_proto, map_res):
        input_file = open(path_to_pgm, 'r')
        output_file = open(path_to_proto, 'w')

        # extract input folder from inputFile
        input_folder = os.path.dirname(input_file.name)

        # read yaml file
        with open(input_file.name) as yaml_file:
            map_properties = yaml.load(yaml_file, Loader=yaml.BaseLoader)

        imagefile = input_folder + '/' + map_properties.get('image')
        # map_res = float(map_properties.get('map_res', .05))
        # origin = map_properties.get('origin', [0, 0, 0])
        origin = [-30.0*map_res, 0, 0] #TODO: add proper fix
        
        make_float = lambda x: -float(x)
        origin = [make_float(origin[0]), make_float(origin[1]), make_float(origin[2])]
        occupied_thresh = 255 * (1 - float(map_properties.get('occupied_thresh', 0.65)))
        free_thresh = 255 * (1 - float(map_properties.get('free_thresh', 0.196)))

        # read pixels from pgm image
        pgm_file = open(imagefile, 'rb')
        map_raster = self.read_pgm(pgm_file)
        pgm_file.close()

        coords = []
        indices = []
        image_height = len(map_raster)

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
                        i = self.add_vertex_points(points, index, i, c, r, c+1, r, image_height, origin, map_res)
                    # free space below pixel?
                    if map_raster[r+1][c] > free_thresh:
                        i = self.add_vertex_points(points, index, i, c, r+1, c+1, r+1, image_height, origin, map_res)
                    # free space left pixel?
                    if map_raster[r][c-1] > free_thresh:
                        i = self.add_vertex_points(points, index, i, c, r, c, r+1, image_height, origin, map_res)
                    # free space right pixel?
                    if map_raster[r][c+1] > free_thresh:
                        i = self.add_vertex_points(points, index, i, c+1, r, c+1, r+1, image_height, origin, map_res)
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
        pf.write('PROTO ' + file_name + ' [\n')
        pf.write('  field  SFVec3f     translation     0 0 0\n')
        pf.write('  field  SFRotation  rotation        1 0 0 1.5708\n')
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

    def add_to_webots_world(self, file_name_list, world_file_dir):
        """ Adds externproto webots import statements to the world.wbt file"""

        for extended_file_name in os.listdir(world_file_dir):
            if extended_file_name.endswith('.wbt'):
                path_to_world = os.path.join(world_file_dir, extended_file_name)

                wb_world_file = open(path_to_world, 'r')
                replaced_content = ''
                i = 0
                for line in wb_world_file:
                    new_line = ''
                    if  i==0 and 'IMPORTABLE EXTERNPROTO' in line and 'robot.proto' not in line:
                        for file_name in file_name_list:
                            import_line = 'IMPORTABLE EXTERNPROTO "../protos/converted_maps/' + file_name + '.proto"' + '\n'
                            new_line += import_line
                        i+=1
                    elif 'IMPORTABLE EXTERNPROTO' in line and 'robot.proto' not in line and 'OilBarrel' not in line:
                        new_line = ''
                    else:
                        new_line = line

                    replaced_content = replaced_content + new_line
                wb_world_file.close()

                write_file = open(path_to_world, 'w')
                write_file.write(replaced_content)
                write_file.close()

    def delete_old_protos(self, map_proto_dir):
        for extended_file_name in os.listdir(map_proto_dir):
            file_path = os.path.join(map_proto_dir, extended_file_name)
            os.remove(file_path)

    def convert(self, map_res):
        self.delete_old_protos(self.map_proto_dir)

        file_name_list = []
        yaml_extension = '.yaml'

        # get filenames (prevent duplicates as there are .yaml as well as .pgms)
        for extended_file_name in os.listdir(self.map_dir):
            if extended_file_name.endswith(yaml_extension):
                file_name = extended_file_name.replace(yaml_extension, '')
                file_name_list.append(file_name)

        # convert maps to protos
        for file_name in file_name_list:
            path_to_pgm = os.path.join(self.map_dir, file_name + yaml_extension)
            path_to_proto = os.path.join(self.map_proto_dir, file_name + '.proto')
            self.convert_pgm_to_proto(file_name, path_to_pgm, path_to_proto, map_res)

        self.add_to_webots_world(file_name_list, self.world_file_dir)
        print('map2proto was succesfull')