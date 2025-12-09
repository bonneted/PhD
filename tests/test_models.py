
import unittest
import sys
import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from phd.models.cm.analytical_plate import train
from phd.plot.plot_cm import compute_metrics_from_history, animate

class TestModels(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests_output")
        self.test_dir.mkdir(exist_ok=True)
        
    def tearDown(self):
        # Cleanup after tests
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_analytical_plate_forward(self):
        print("\nTesting Analytical Plate Forward...")
        config = {
            "task": "forward",
            "n_iter": 10,
            "log_every": 2,
            "results_dir": str(self.test_dir),
            "generate_video": True,
            "net_type": "SPINN",
            "available_time": None
        }
        results = train(config)
        
        run_dir = Path(results["run_dir"])
        self.assertTrue(run_dir.exists())
        
        # Check video
        video_path = run_dir / "training_animation.mp4"
        self.assertTrue(video_path.exists(), "Video file was not generated")
        self.assertGreater(video_path.stat().st_size, 0, "Video file is empty")
        
        # Check data loading
        data = load_run_data(run_dir)
        self.assertIn("fields", data)
        self.assertIn("Ux", data["fields"])

    def test_analytical_plate_inverse(self):
        print("\nTesting Analytical Plate Inverse...")
        config = {
            "task": "inverse",
            "n_iter": 10,
            "log_every": 2,
            "results_dir": str(self.test_dir),
            "generate_video": True,
            "net_type": "SPINN",
            "n_DIC": 10,
            "lmbd_init": 2.0,
            "mu_init": 0.3
        }
        results = train(config)
        run_dir = Path(results["run_dir"])
        
        self.assertTrue((run_dir / "variables.dat").exists())
        
        # Check video
        video_path = run_dir / "training_animation.mp4"
        self.assertTrue(video_path.exists(), "Video file was not generated")
        
        # Check data loading
        data = load_run_data(run_dir)
        self.assertIn("variable_values", data)
        self.assertEqual(data["variable_values"].shape[1], 2)

if __name__ == '__main__':
    unittest.main()
