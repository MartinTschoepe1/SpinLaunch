#!/usr/bin/env python3

import pathlib
import sys
import unittest as ut

import numpy as np

sys.path.insert(
    0, str(
        pathlib.Path(__file__).resolve().parent.parent.joinpath('solutions')))
import ex_2_1  # isort:skip


class Tests(ut.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gravity = 9.81
        cls.mass = 42.0
        cls.v = np.array([0.1, 0.2])
        cls.x = np.zeros(2)
        cls.dt = 0.1
        cls.f = np.array([0.0, -412.02000000000004])

    def test_force(self):
        np.testing.assert_array_equal(
            ex_2_1.force(
                self.mass,
                self.gravity),
            self.f,
            err_msg="Implementation of the function 'force' seems wrong.")

    def test_step_euler(self):
        x, v = ex_2_1.step_euler(
            self.x, self.v, self.dt, self.mass, self.gravity, self.f)
        np.testing.assert_almost_equal(x, np.array(
            [0.01, 0.02]), err_msg="Position is wrong after integration.")
        np.testing.assert_almost_equal(v, np.array(
            [0.1, -0.7810000000000001]), err_msg="Velocity is wrong after integration.")


if __name__ == "__main__":
    ut.main()
