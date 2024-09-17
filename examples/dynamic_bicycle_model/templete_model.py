#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
import os

rel_do_mpc_path = os.path.join('..', '..')
sys.path.append(rel_do_mpc_path)
import do_mpc


def template_model(symvar_type='SX'):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """

    "Chronos and CRS: Design of a miniature car-like robot and a software framework for single and multi-agent robotics and control"
    model_type = 'continuous'  # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    # Certain parameters
    m = 2.0  # Mass [kg]
    lf = 0.3  # Distance from CoG to front wheel [m]
    lr = 0.3  # Distance from CoG to rear wheel [m]
    w = 0.15  # Width of the car [m]
    D_f = 0.5  # Peak factor for front wheel between 0.5 and 1.5
    D_r = 0.5  # Peak factor for rear wheel between 0.5 and 1.5
    C_f = 1  # shape factor for front wheel between 1 and 2
    C_r = 1  # shape factor for rear wheel between 1 and 2
    B_f = 5  # stiffness factor for front wheel between 5 and 15
    B_r = 5  # stiffness factor for rear wheel between 5 and 15
    # E_f = 0.5  # curvature factor for front wheel between 0.5 and 1.0
    # E_r = 0.5  # curvature factor for rear wheel between 0.5 and 1.0

    I_z = (1 / 12) * m * ((lf + lr) ** 2 + w ** 2)  # Moment of inertia around z-axis [kg*m^2]
    c_1 = 0.5  # acceleration constant between 0.1 and 1
    c_2 = 0.1  # velocity related coefficient between 0.01 and 0.1
    c_3 = 0.01  # coefficient related to velocity squared between 0.001 and 0.01
    c_4 = 0.01  # constant force offset between  0.01 and 0.1

    # States struct (optimization variables):
    X_p = model.set_variable(var_type='_x', var_name='X_p', shape=(1, 1))
    Y_p = model.set_variable(var_type='_x', var_name='Y_p', shape=(1, 1))
    Psi = model.set_variable(var_type='_x', var_name='Psi', shape=(1, 1))
    V_x = model.set_variable(var_type='_x', var_name='V_x', shape=(1, 1))
    V_y = model.set_variable(var_type='_x', var_name='V_y', shape=(1, 1))
    W = model.set_variable(var_type='_x', var_name='W', shape=(1, 1))

    # Input struct (optimization variables):
    Delta = model.set_variable(var_type='_u', var_name='Delta')
    Acc = model.set_variable(var_type='_u', var_name='Acc')


    # # Set expression. These can be used in the cost function, as non-linear constraints
    # # or just to monitor another output.
    Vel = model.set_expression(expr_name='Vel', expr=sqrt(V_x ** 2 + V_y ** 2))

    # Expressions can also be formed without beeing explicitly added to the model.
    # The main difference is that they will not be monitored and can only be used within the current file.
    alpha_f = atan((V_y + lf * W) / V_x) - Delta  # Front wheel slip angle
    alpha_r = atan((V_y - lr * W) / V_x)  # Rear wheel slip angle
    F_f = D_f * sin(C_f * atan(B_f * alpha_f))  # Front wheel force
    F_r = D_r * sin(C_r * atan(B_r * alpha_r))  # Rear wheel force
    F_x = (c_1 + c_2 * V_x) * Acc + c_3 * V_x ** 2 + c_4  # Force in x-direction

    # Differential equations
    model.set_rhs('X_p', V_x * cos(Psi) - V_y * sin(Psi))
    model.set_rhs('Y_p', V_x * sin(Psi) + V_y * cos(Psi))
    model.set_rhs('Psi', W)
    model.set_rhs('V_x', (1 / m) * (F_x - F_f * sin(Delta) + m * V_y * W))
    model.set_rhs('V_y', (1 / m) * (F_r + F_f * cos(Delta) - m * V_x * W))
    model.set_rhs('W', (1 / I_z) * (F_f * lf * cos(Delta) - F_r * lr))

    # Build the model
    model.setup()

    return model
