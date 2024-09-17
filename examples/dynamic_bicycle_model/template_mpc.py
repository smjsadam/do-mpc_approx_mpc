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


def template_mpc(model, silence_solver=False):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    # Set settings of MPC:
    mpc.settings.n_horizon = 20
    mpc.settings.n_robust = 0
    mpc.settings.open_loop = 0
    mpc.settings.t_step = 0.1
    mpc.settings.state_discretization = 'collocation'
    mpc.settings.collocation_type = 'radau'
    mpc.settings.collocation_deg = 2
    mpc.settings.collocation_ni = 1
    mpc.settings.store_full_solution = True

    if silence_solver:
        mpc.settings.supress_ipopt_output()

    mterm = (model.x['X_p'] - 1) ** 2
    lterm = (model.x['Y_p'] - 1) ** 2

    mpc.set_objective(mterm=mterm, lterm=lterm)

    mpc.set_rterm(Delta=0.1, Acc=1e-3)

    # Set constraints:
    mpc.bounds['lower', '_x', 'X_p'] = -5
    mpc.bounds['lower', '_x', 'Y_b'] = -5
    mpc.bounds['lower', '_x', 'Psi'] = -pi / 3
    mpc.bounds['lower', '_x', 'V_x'] = -5
    mpc.bounds['lower', '_x', 'V_y'] = -5
    mpc.bounds['lower', '_x', 'W'] = -50

    mpc.bounds['upper', '_x', 'X_p'] = 5
    mpc.bounds['upper', '_x', 'Y_b'] = 5
    mpc.bounds['upper', '_x', 'Psi'] = pi / 3
    mpc.bounds['upper', '_x', 'V_x'] = 5
    mpc.bounds['upper', '_x', 'V_y'] = 5
    mpc.bounds['upper', '_x', 'W'] = 50

    mpc.bounds['lower', '_u', 'Delta'] = -5
    mpc.bounds['lower', '_u', 'Acc'] = -5

    mpc.bounds['upper', '_u', 'Delta'] = 5
    mpc.bounds['upper', '_u', 'ACC'] = 5


    #mpc.set_nl_cons('T_R', model.x['T_R'], ub=140, soft_constraint=True, penalty_term_cons=1e2)


    mpc.setup()

    return mpc
