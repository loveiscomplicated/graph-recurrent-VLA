"""Run the existing baseline and connectivity contract tests in the remote env."""
from run import *
import unittest
suite=unittest.defaultTestLoader.loadTestsFromNames(['test_optimization','test_radius_components'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
