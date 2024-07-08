import unittest
import numpy as np
from unittest.mock import MagicMock
from environments.base_env import BaseEnv   # Replace with the actual class name and module

class TestYourClassName(unittest.TestCase):
    def setUp(self):
        self.env = BaseEnv()
        self.env.cur_pos = np.array([0, 0])
        self.env.path = [np.array([0, 0]), np.array([10, 0])]
        self.env.initial_progress = 0
        self.env.goal_progress = 10
        self.env.direction = 1

    def test_update_path_dist_and_path_progress(self):
        self.env.update_path_dist_and_path_progress()
        self.assertEqual(self.env.path_dist, 0)
        self.assertEqual(self.env.path_progress, 0)

    def test_reset_path_progress(self):
        self.env.reset_path_progress()
        self.assertEqual(self.env.path_progress, 0)
        self.assertEqual(self.env.prev_path_progress, 0)
        # Assuming calculate_goal_progress and calculate_initial_progress are correctly implemented
        self.assertTrue(isinstance(self.env.goal_progress, float))
        self.assertTrue(isinstance(self.env.initial_progress, float))

    def test_calculate_goal_progress(self):
        progress = self.env.calculate_goal_progress()
        self.assertEqual(progress, 10)

    def test_calculate_initial_progress(self):
        progress = self.env.calculate_initial_progress()
        self.assertEqual(progress, 0)

    def test_reset_map_path_and_poses(self):
        self.env.train_map_nr_list = [1, 2, 3]
        self.env.grids_dir = '/path/to/grids'
        self.env.paths_dir = '/path/to/paths'
        self.env.params = {'map_res': 1}
        # Mocking external dependencies
        np.load = MagicMock(return_value=np.array([[0, 0], [10, 0]]))
        self.env.reset_map_path_and_poses()
        self.assertTrue(isinstance(self.env.map_nr, int))
        self.assertTrue(isinstance(self.env.grid, np.ndarray))
        self.assertTrue(isinstance(self.env.path, np.ndarray))
        self.assertTrue(isinstance(self.env.init_pose, np.ndarray))
        self.assertTrue(isinstance(self.env.goal_pose, np.ndarray))
        self.assertTrue(isinstance(self.env.direction, int))

if __name__ == '__main__':
    unittest.main()