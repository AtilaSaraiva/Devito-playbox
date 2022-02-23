from devito import Eq, TimeFunction, sqrt, Function, Operator, Grid, solve
from matplotlib import pyplot as plt
import numpy as np

def Shallow_water_2D(eta0, M0, N0, h0, grid, g, alpha, nt, dx, dy, dt):
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

    eta.data[0] = eta0.copy()
    M.data[0]   = M0.copy()
    N.data[0]   = N0.copy()
    D.data[:]   = D0.copy()
    h.data[:]   = h0.copy()

    frictionTerm = g * alpha**2 * sqrt(M**2 + N**2 ) / D**(7./3.)

    pde_eta = Eq(eta.dt + M.dxc + N.dyc)
    pde_M   = Eq(M.dt + (M**2/D).dxc + (M*N/D).dyc + g*D*eta.forward.dxc + frictionTerm*M)
    pde_N   = Eq(N.dt + (M*N/D).dxc + (N**2/D).dyc + g*D*eta.forward.dyc + frictionTerm*N)

    # Defining boundary conditions
    x, y = grid.dimensions
    t = grid.stepping_dim
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

    optime = Operator([update_eta, bc_left, bc_right, bc_top, bc_bottom,
                       update_M, update_N, eq_D])

    optime(time=nt, dt=dt)
    return eta, M, N

Lx    = 100.0   # width of the mantle in the x direction []
Ly    = 100.0   # thickness of the mantle in the y direction []
nx    = 401     # number of points in the x direction
ny    = 401     # number of points in the y direction
dx    = Lx / (nx - 1)  # grid spacing in the x direction []
dy    = Ly / (ny - 1)  # grid spacing in the y direction []
g     = 9.81  # gravity acceleration [m/s^2]
alpha = 0.025 # friction coefficient for natural channels in good condition
# Maximum wave propagation time [s]
Tmax  = 1.
dt    = 1/4500.
nt    = (int)(Tmax/dt)
print(dt, nt)

x = np.linspace(0.0, Lx, num=nx)
y = np.linspace(0.0, Ly, num=ny)

# Define initial eta, M, N
X, Y = np.meshgrid(x,y) # coordinates X,Y required to define eta, h, M, N

# Define constant ocean depth profile h = 50 m
h0 = 50. * np.ones_like(X)

# Define initial eta Gaussian distribution [m]
eta0 = 0.5 * np.exp(-((X-50)**2/10)-((Y-50)**2/10))

# Define initial M and N
M0 = 100. * eta0
N0 = 0. * M0
D0 = eta0 + 50.

grid  = Grid(shape=(ny, nx), extent=(Ly, Lx))

eta, M, N = Shallow_water_2D(eta0, M0, N0, h0, grid, g, alpha, nt, dx, dy, dt)


print(eta.data)
plt.imshow(eta.data[1])
plt.savefig('out.pdf')
