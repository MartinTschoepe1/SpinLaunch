#!/usr/bin/env python3

import pathlib
import sys
import unittest as ut

import numpy as np

sys.path.insert(
    0, str(
        pathlib.Path(__file__).resolve().parent.parent.joinpath('solutions')))
import ex_2_1  # isort:skip
import ex_2_2  # isort:skip


class Test(ut.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mass = 21
        cls.gravity = 9.81
        cls.v = np.array([0.1, 0.2])
        cls.gamma = 2.5
        cls.v_0 = np.array([2.1, 1.3])

    def test_force(self):
        np.testing.assert_almost_equal(ex_2_2.force(self.mass, self.gravity, self.v, self.gamma, self.v_0),
                                       ex_2_1.force(self.mass, self.gravity) - np.array([-5., -2.75]), err_msg="Force calculation with wind is not correct.")


if __name__ == "__main__":
    ut.main()
