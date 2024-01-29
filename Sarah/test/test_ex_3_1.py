#!/usr/bin/env python3

import os
import pathlib
import sys
import unittest as ut

import numpy as np
import scipy.constants

sys.path.insert(
    0, str(
        pathlib.Path(__file__).resolve().parent.parent.joinpath('solutions')))
import ex_3_1  # isort:skip


class Tests(ut.TestCase):
    @classmethod
    def setUpClass(cls):
        current_dir = pathlib.Path(__file__).resolve()
        data = np.load(current_dir.parent.parent.joinpath(
            'files/solar_system.npz'))
        cls.x_init = data['x_init']
        cls.v_init = data['v_init']
        cls.masses = data['m']
        cls.g = data['g']

    def test_step_euler_force(self):
        x = self.x_init.copy()
        v = self.v_init.copy()
        dt = 1e-4
        forces_expected = np.array([[565.96044578, -39.19537202, -0.71291931, -1.69684599, -61.46155796, -462.8937505],
                                    [0., 0., 0., 0., 0., 0.]])
        x_new_expected = np.array([[0.00000000e+00, 1.00000000e+00, 1.00256267e+00, 1.52410000e+00, 7.23000000e-01, 5.20300000e+00],
                                   [0.00000000e+00, 6.28318531e-04, 6.49535585e-04, 5.23250598e-04, 7.40223744e-04, 2.75644293e-04]])
        v_new_expected = np.array([[1.69985657e-07, -3.91953720e-03, -5.72474203e-03, -1.69684599e-03, -7.54100118e-03, -1.45620182e-04],
                                   [0.00000000e+00, 6.28318531e+00, 6.49535585e+00, 5.23250598e+00, 7.40223744e+00, 2.75644293e+00]])
        x_new, v_new = ex_3_1.step_euler(
            x, v, dt, self.masses, self.g, ex_3_1.forces)
        np.testing.assert_almost_equal(
            x_new,
            x_new_expected,
            err_msg="Positions are wrong after integration.")
        np.testing.assert_almost_equal(
            v_new,
            v_new_expected,
            err_msg="Velocities are wrong after integration.")
        np.testing.assert_almost_equal(
            ex_3_1.forces(
                self.x_init,
                self.masses,
                self.g),
            forces_expected, err_msg="The function 'forces' does not work as expected.")

    def test_gravitational_force(self):
        r_1 = self.x_init[:, 0]
        r_2 = self.x_init[:, 1]
        r_12 = r_2 - r_1
        m_1 = self.masses[0]
        m_2 = self.masses[1]
        np.testing.assert_almost_equal(ex_3_1.force(
            r_12, m_1, m_2, self.g), np.array([-39.4208064, 0.]), err_msg="The function 'force' does not work as expected.")


if __name__ == "__main__":
    ut.main()
