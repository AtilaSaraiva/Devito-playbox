from devito import Eq, TimeFunction, sqrt, Function, Operator, Grid, solve, ConditionalDimension
from matplotlib import pyplot as plt
import numpy as np
from devito.tools import memoized_meth


class ShallowWaterWaveSolver ():
    def __init__(self, gravity, alpha, grid, **kwargs):
        self.gravity = gravity
        self.alpha = alpha
        self.grid = grid
        self.eta   = TimeFunction(name='eta', grid=self.grid, space_order=2)
        self.M     = TimeFunction(name='M', grid=self.grid, space_order=2)
        self.N     = TimeFunction(name='N', grid=self.grid, space_order=2)
        self.h     = Function(name='h', grid=self.grid)
        self.D     = Function(name='D', grid=self.grid)

    @memoized_meth
    def op_shallowWaver(self, etasave):
        eta   = self.eta
        M     = self.M
        N     = self.N
        h     = self.h
        D     = self.D
        g     = self.gravity
        alpha = self.alpha

        frictionTerm = g * alpha**2 * sqrt(M**2 + N**2 ) / D**(7./3.)

        pde_eta = Eq(eta.dt + M.dxc + N.dyc)
        pde_M   = Eq(M.dt + (M**2/D).dxc + (M*N/D).dyc + g*D*eta.forward.dxc + frictionTerm*M)
        pde_N   = Eq(N.dt + (M*N/D).dxc + (N**2/D).dyc + g*D*eta.forward.dyc + frictionTerm*N)

        # Defining boundary conditions
        x, y = self.grid.dimensions
        t = self.grid.stepping_dim
        bc_left   = Eq(eta[t+1, 0, y], eta[t+1, 1, y])
        bc_right  = Eq(eta[t+1, nx-1, y], eta[t+1, nx-2, y])
        bc_top    = Eq(eta[t+1, x, 0], eta[t+1, x, 1])
        bc_bottom = Eq(eta[t+1, x, ny-1], eta[t+1, x, ny-2])

        stencil_eta = solve(pde_eta, eta.forward)
        stencil_M   = solve(pde_M, M.forward)
        stencil_N   = solve(pde_N, N.forward)

        update_eta  = Eq(eta.forward, stencil_eta, subdomain=grid.interior)
        update_M    = Eq(M.forward, stencil_M, subdomain=grid.interior)
        update_N    = Eq(N.forward, stencil_N, subdomain=grid.interior)
        eq_D        = Eq(D, eta.forward + h)

        return Operator([update_eta, bc_left, bc_right, bc_top, bc_bottom,
                           update_M, update_N, eq_D] + [Eq(etasave, eta)])


    def forward(eta0, M0, N0, h0, nt, dt, grid=None, alpha=None, g=None, nsnaps=100):
        if grid:
            self.eta   = TimeFunction(name='eta', grid=grid, space_order=2)
            self.M     = TimeFunction(name='M', grid=grid, space_order=2)
            self.N     = TimeFunction(name='N', grid=grid, space_order=2)
            self.h     = Function(name='h', grid=grid)
            self.D     = Function(name='D', grid=grid)
            self.grid  = grid

        self.alpha = alpha or self.alpha

        self.eta.data[0] = eta0.copy()
        self.M.data[0]   = M0.copy()
        self.N.data[0]   = N0.copy()
        self.D.data[:]   = eta0 + h0
        self.h.data[:]   = h0.copy()

        factor = round(nt / nsnaps)
        time_subsampled = ConditionalDimension('t_sub', parent=grid.time_dim, factor=factor)
        etasave = TimeFunction(name='etasave', grid=grid, space_order=2,
                              save=nsnaps, time_dim=time_subsampled)


        self.op_shallowWaver(eta=self.eta, M=self.M, N=self.N, D=self.D, h=self.h, nt=nt-2, dt=dt)

        return etasave, M, N






def Shallow_water_2D(eta0, M0, N0, h0, grid, g, alpha, nt, dx, dy, dt, nsnaps=100):
    """
    Computes and returns the discharge fluxes M, N and wave height eta from
    the 2D Shallow water equation using the FTCS finite difference method.

    Parameters
    ----------
    eta0 : numpy.ndarray
        The initial wave height field as a 2D array of floats.
    M0 : numpy.ndarray
        The initial discharge flux field in x-direction as a 2D array of floats.
    N0 : numpy.ndarray
        The initial discharge flux field in y-direction as a 2D array of floats.
    h : numpy.ndarray
        Bathymetry model as a 2D array of floats.
    g : float
        gravity acceleration.
    alpha : float
        Manning's roughness coefficient.
    nt : integer
        Number fo timesteps.
    dx : float
        Spatial gridpoint distance in x-direction.
    dy : float
        Spatial gridpoint distance in y-direction.
    dt : float
        Time step.
    """

    eta   = TimeFunction(name='eta', grid=grid, space_order=2)
    M     = TimeFunction(name='M', grid=grid, space_order=2)
    N     = TimeFunction(name='N', grid=grid, space_order=2)
    h     = Function(name='h', grid=grid)
    D     = Function(name='D', grid=grid)

    factor = round(nt / nsnaps)
    print(factor, nt, grid.time_dim)
    time_subsampled = ConditionalDimension(
        't_sub', parent=grid.time_dim, factor=factor)
    etasave = TimeFunction(name='etasave', grid=grid, space_order=2,
                         save=nsnaps, time_dim=time_subsampled)


    eta.data[0] = eta0.copy()
    M.data[0]   = M0.copy()
    N.data[0]   = N0.copy()
    D.data[:]   = D0.copy()
    h.data[:]   = h0.copy()


    optime(time=nt-2, dt=dt)
    return etasave, M, N
