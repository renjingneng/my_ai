import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import my_ai


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_basic(self):
        temp = my_ai.utility.Accumulator(2)
        self.assertEqual(temp[0], 0)


if __name__ == '__main__':
    unittest.main()