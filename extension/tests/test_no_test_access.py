import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
DEV_ONLY = ("dev_c1.py", "dev_c2.py", "dev_c3c4.py")


class TestDevScriptsNeverReadTest(unittest.TestCase):
    def test_dev_scripts_never_read_test(self):
        self.assertTrue((SCRIPTS / "dev_c1.py").exists(), "dev_c1.py must exist")
        for name in DEV_ONLY:
            path = SCRIPTS / name
            if path.exists():
                self.assertNotIn("final_test", path.read_text(encoding="utf-8"), f"{name} must not read the test split")


if __name__ == "__main__":
    unittest.main()
