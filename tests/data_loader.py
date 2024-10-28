import pytest
import numpy as np
from io import BytesIO
from utils.data_tools import read_pgm, add_vertex_points, convert_pgm_to_proto

def test_read_pgm():
    pgm_data = b"P5\n4 4\n255\n" + bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    pgmf = BytesIO(pgm_data)
    raster = read_pgm(pgmf)
    expected_raster = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ]
    assert raster == expected_raster

def test_add_vertex_points():
    points = []
    index = []
    i = 0
    x1, y1, x2, y2 = 1, 1, 2, 2
    image_height = 4
    origin = [0, 0]
    map_res = 1.0
    new_i = add_vertex_points(points, index, i, x1, y1, x2, y2, image_height, origin, map_res)
    expected_points = [
        [1, 0, 2],
        [2, 0, 2],
        [2, 1, 2],
        [1, 1, 2]
    ]
    expected_index = [
        [0, 1, 2, 3, -1],
        [3, 2, 1, 0, -1]
    ]
    assert points == expected_points
    assert index == expected_index
    assert new_i == 4

def test_convert_pgm_to_proto(tmp_path):
    class MockCfg:
        resolution = 1.0
        occupied_thresh = 0.65
        free_tresh = 0.196

    map_cfg = MockCfg()
    map_raster = [
        [255, 255, 255, 255],
        [255, 0, 0, 255],
        [255, 0, 0, 255],
        [255, 255, 255, 255]
    ]
    output_file = tmp_path / "output.proto"
    convert_pgm_to_proto(map_cfg, map_raster, output_file)
    with open(output_file, 'r') as f:
        content = f.read()
    assert "PROTO" in content
    assert "Coordinate" in content
    assert "IndexedFaceSet" in content