import unittest
# right now this is just so a test is there as a framework, no actual test happening yet


class TestTFM(unittest.TestCase):
    def test_upper(self):
        # test if the test works with a simple sum
        sum = 3 + 3
        self.assertEqual(sum, 6)

    def hor_dist(self):
        self.assertEqual(0, 0)


if __name__ == "__main__":
    unittest.main()
