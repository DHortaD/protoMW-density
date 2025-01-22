import numpy as np
from galpy.util import coords, _rotate_to_arbitrary_vector
from scipy.optimize import newton
from scipy.special import erfinv

_R0 = 8.275 # Gravity Collab.
_z0 = 0.02 #bennett and bovy

#utilities for transformations etc


def transform_abg(xyz,delta,beta,gamma):
    """
    Transform xyz coordinates by rotation around x-axis (gamma), transformed y-axis (beta) and twice transformed z-axis (delta)
    """
    Rx = np.zeros([3,3])
    Ry = np.zeros([3,3])
    Rz = np.zeros([3,3])
    Rx[0,0] = 1
    Rx[1] = [0, np.cos((gamma)), -np.sin((gamma))]
    Rx[2] = [0, np.sin((gamma)), np.cos((gamma))]
    Ry[0] = [np.cos((beta)), 0, np.sin((beta))]
    Ry[1,1] = 1
    Ry[2] = [-np.sin((beta)), 0, np.cos((beta))]
    Rz[0] = [np.cos((delta)), -np.sin((delta)), 0]
    Rz[1] = [np.sin((delta)), np.cos((delta)), 0]
    Rz[2,2] = 1
    R = np.matmul(Rx,np.matmul(Ry,Rz))
    if np.ndim(xyz) == 1:
        tgalcenrect = np.dot(R, xyz)
        x, y, z = tgalcenrect[0], tgalcenrect[1], tgalcenrect[2]
    else:
        tgalcenrect = np.einsum('ij,aj->ai', R, xyz)
        x, y, z = tgalcenrect[:,0], tgalcenrect[:,1], tgalcenrect[:,2]
    return x, y, z


def transform_aby(xyz,alpha,beta,gamma):
    """
    Transform xyz coordinates by rotation around x-axis (alpha), transformed y-axis (beta) and twice transformed z-axis (gamma)
    """
    Rx = np.zeros([3,3])
    Ry = np.zeros([3,3])
    Rz = np.zeros([3,3])
    Rx[0,0] = 1
    Rx[1] = [0, np.cos(alpha), -np.sin(alpha)]
    Rx[2] = [0, np.sin(alpha), np.cos(alpha)]
    Ry[0] = [np.cos(beta), 0, np.sin(beta)]
    Ry[1,1] = 1
    Ry[2] = [-np.sin(beta), 0, np.cos(beta)]
    Rz[0] = [np.cos(gamma), -np.sin(gamma), 0]
    Rz[1] = [np.sin(gamma), np.cos(gamma), 0]
    Rz[2,2] = 1
    R = np.matmul(Rx,np.matmul(Ry,Rz))
    if np.ndim(xyz) == 1:
        tgalcenrect = np.dot(R, xyz)
        x, y, z = tgalcenrect[0], tgalcenrect[1], tgalcenrect[2]
    else:
        tgalcenrect = np.einsum('ij,aj->ai', R, xyz)
        x, y, z = tgalcenrect[:,0], tgalcenrect[:,1], tgalcenrect[:,2]
    return x, y, z

def transform_zvecpa(xyz,zvec,pa):
    """
    transform coordinates using the axis-angle method
    """
    pa_rot= np.array([[np.cos(pa),np.sin(pa),0.],
                         [-np.sin(pa),np.cos(pa),0.],
                         [0.,0.,1.]])

    zvec/= np.sqrt(np.sum(zvec**2.))
    zvec_rot= _rotate_to_arbitrary_vector(np.array([[0.,0.,1.]]),zvec,inv=True)[0]
    trot= np.dot(pa_rot,zvec_rot)
    if np.ndim(xyz) == 1:
        tgalcenrect = np.dot(trot, xyz)
        x, y, z = tgalcenrect[0], tgalcenrect[1], tgalcenrect[2]
    else:
        tgalcenrect = np.einsum('ij,aj->ai', trot, xyz)
        x, y, z = tgalcenrect[:,0], tgalcenrect[:,1], tgalcenrect[:,2]
    return x, y, z

############################ density models

# SPL

def spherical(R,phi,z,params=[2.5]):
    """
    general spherical power-law density model
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [alpha,]
    OUTPUT
        density at R, phi, z (normalised to 1 at R=8 kpc and z=0.02)
    """
    grid = False
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    dens = np.sqrt(x**2+y**2+z**2)**(-params[0])
    dens = dens/(np.sqrt(_R0**2+_z0**2)**(-params[0]))
    if grid:
        dens = dens.reshape(dim)
    return dens

def axisymmetric(R,phi,z,params=[2.5,1.]):
    """
    general axisymmetric power-law density model
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [alpha,q]
    OUTPUT
        density at R, phi, z (normalised to 1 at R=8 kpc and z=0.02)
    """
    grid = False
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    dens = np.sqrt(x**2+y**2+z**2/params[1]**2)**-params[0]
    dens = dens/np.sqrt(_R0**2+_z0**2/params[1]**2)**-params[0]
    if grid:
        dens = dens.reshape(dim)
    return dens

def triaxial(R,phi,z,params=[2.5, 1., 1.]):
    """
    general axisymmetric power-law density model
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [alpha,q, p]
    OUTPUT
        density at R, phi, z (normalised to 1 at R=8 kpc and z=0.02)
    """
    grid = False
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    dens = np.sqrt(x**2+y**2/params[2]**2+z**2/params[1]**2)**-params[0]
    dens = dens/np.sqrt(_R0**2+_z0**2/params[1]**2)**-params[0]
    if grid:
        dens = dens.reshape(dim)
    return dens


# exponential disc

def justexpdisk(R,phi,z,params=[1/1.8,1/0.8]):
    """
    vanilla exponential disk
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [1/hr, 1/hz]
    OUTPUT
        density at R, phi, z
    """
    hr = params[0]
    hz = params[1]
    diskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(z))
    diskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(_z0))
    return diskdens/diskdens_sun

def justexpdisk_fixed(R,phi,z):
    """
    vanilla exponential disk
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [1/hr, 1/hz]
    OUTPUT
        density at R, phi, z
    """
    hr = 1./2.2
    hz = 1./0.8
    diskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(z))
    diskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(_z0))
    return diskdens/diskdens_sun

    
# Plummer
def plummer(R, phi, z, params = [3.]):
    
    """
    vanilla plummer profile
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [a] (size of core)
    OUTPUT
        density at R, phi, z
    """
       
    a = params[0]
    r_all = np.sqrt(R**2. + z**2.)
    r_sun = np.sqrt(_R0**2. + _z0**2.)
    dens = (3./4.*np.pi*a**3.) * (1. + (r_all**2./a**2.))**(-5./2.)
    dens_sun = (3./4.*np.pi*a**3.) * (1. + (r_sun**2./a**2.))**(-5./2.)
    return dens/dens_sun


def plummer_axi(R, phi, z, params = [3., 0., 0.,]):
    
    """
    vanilla plummer profile
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [a] (size of core)
    OUTPUT
        density at R, phi, z
    """
       
    a = params[0]
    p = params[1]
    q = params[2]
    
    # Convert from cylindrical coords to cartesian
    x,y,z = R*np.cos(phi), R*np.sin(phi), z
    x_sun, y_sun, z_sun = _R0, 0., _z0
   
    # Define radius with transformed coordinate reference frame
    r_all = np.sqrt(x**2+(y**2/p**2)+(z**2/q**2))
    r_sun = np.sqrt(x_sun**2+(y_sun**2/p**2)+(z_sun**2/q**2))
    
    dens = (3./4.*np.pi*a**3.) * (1. + (r_all**2./a**2.))**(-5./2.)
    dens_sun = (3./4.*np.pi*a**3.) * (1. + (r_sun**2./a**2.))**(-5./2.)
    return dens/dens_sun
    
# Hernquist
def hernquist(R, phi, z, params = [3.]):
    
    """
    vanilla Hernquist profile
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [a] (size of core)
    OUTPUT
        density at R, phi, z
    """   
    a = params[0]
    r_all = np.sqrt(R**2. + z**2.)
    r_sun = np.sqrt(_R0**2. + _z0**2.)    
    
    dens = (a/2.*np.pi*r_all) * (1./(r_all+a)**3.) 
    dens_sun = (a/2.*np.pi*r_sun) * (1./(r_sun+a)**3.) 
    return dens/dens_sun

# Hernquist
def hernquist_axi(R, phi, z, params = [3., 0.,0.]):
    
    """
    vanilla Hernquist profile
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [a] (size of core)
    OUTPUT
        density at R, phi, z
    """   
    a = params[0]
    p = params[1]
    q = params[2]
    
    # Convert from cylindrical coords to cartesian
    x,y,z = R*np.cos(phi), R*np.sin(phi), z
    x_sun, y_sun, z_sun = _R0, 0., _z0
    
    # Define radius with transformed coordinate reference frame
    r_all = np.sqrt(x**2+(y**2/p**2)+(z**2/q**2))
    r_sun = np.sqrt(x_sun**2+(y_sun**2/p**2)+(z_sun**2/q**2))  
    
    dens = (a/2.*np.pi*r_all) * (1./(r_all+a)**3.) 
    dens_sun = (a/2.*np.pi*r_sun) * (1./(r_sun+a)**3.) 
    return dens/dens_sun

# Sersic

def sersic(R, phi, z, params = [5., 0.5, 5.]):
    
    """
    vanilla Sersic profile
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [a] (size of core)
    OUTPUT
        density at R, phi, z
    """  
    
    n = params[0]
    bn = params[1]
    reff = params[2]
    r_all = np.sqrt(R**2. + z**2.)
    r_sun = np.sqrt(_R0**2. + _z0**2.)  
    
    dens = np.exp(-bn * ((r_all/reff)**(1./n) - 1.))
    dens_sun = np.exp(-bn * ((r_sun/reff)**(1./n) - 1.))
        
    return dens/dens_sun

def sersic_axi(R, phi, z, params = [5., 0.5, 5., 0., 0.]):
    
    """
    vanilla Sersic profile
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [a] (size of core)
    OUTPUT
        density at R, phi, z
    """  
    
    n = params[0]
    bn = params[1]
    reff = params[2]
    p = params[3]
    q = params[4]
    
    # Convert from cylindrical coords to cartesian
    x,y,z = R*np.cos(phi), R*np.sin(phi), z
    x_sun, y_sun, z_sun = _R0, 0., _z0    
    
    # Define radius with transformed coordinate reference frame
    r_all = np.sqrt(x**2+(y**2/p**2)+(z**2/q**2))
    r_sun = np.sqrt(x_sun**2+(y_sun**2/p**2)+(z_sun**2/q**2))
    
    dens = np.exp(-bn * ((r_all/reff)**(1./n) - 1.))
    dens_sun = np.exp(-bn * ((r_sun/reff)**(1./n) - 1.))
        
    return dens/dens_sun


# Plummer

def plummer_disc(R, phi, z, params = [3., 0.], split=False):
    
    """
    vanilla plummer profile
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [a] (size of core)
    OUTPUT
        density at R, phi, z
    """
       
    hr = 1/2.2
    hz = 1/0.8
    
    a = params[0]
    fdisc = params[1]
    r_all = np.sqrt(R**2. + z**2.)
    r_sun = np.sqrt(_R0**2. + _z0**2.)
    dens = (3./4.*np.pi*a**3.) * (1. + (r_all**2./a**2.))**(-5./2.)
    sundens = (3./4.*np.pi*a**3.) * (1. + (r_sun**2./a**2.))**(-5./2.)
    
    # disc density (the normalisation constant that measures the mass is removed when normalising by Sun density)
    diskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(z))
    diskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(_z0))
    
    # split or not the total density into halo density and disc density
    if split:
        dens, diskdens = (1-fdisc)*(dens/sundens), (fdisc)*(diskdens/diskdens_sun)
        return dens, diskdens
    else:
        dens = (1-fdisc)*(dens/sundens)+(fdisc*diskdens/diskdens_sun)
        return dens
        
# Hernquist

def hernquist_disc(R, phi, z, params = [3., 0.], split=False):
    
    """
    vanilla Hernquist profile
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [a] (size of core)
    OUTPUT
        density at R, phi, z
    """   
    
    hr = 1/2.2
    hz = 1/0.8
    
    a = params[0]
    fdisc = params[1]
    
    r_all = np.sqrt(R**2. + z**2.)
    r_sun = np.sqrt(_R0**2. + _z0**2.)    
    
    dens = (a/2.*np.pi*r_all) * (1./(r_all+a)**3.) 
    sundens = (a/2.*np.pi*r_sun) * (1./(r_sun+a)**3.) 

    # disc density (the normalisation constant that measures the mass is removed when normalising by Sun density)
    diskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(z))
    diskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(_z0))
    
    # split or not the total density into halo density and disc density
    if split:
        dens, diskdens = (1-fdisc)*(dens/sundens), (fdisc)*(diskdens/diskdens_sun)
        return dens, diskdens
    else:
        dens = (1-fdisc)*(dens/sundens)+(fdisc*diskdens/diskdens_sun)
        return dens
# Sersic

def sersic_disc(R, phi, z, params = [5., 0.5, 5., 0.], split=False):
    
    """
    vanilla Sersic profile
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [a] (size of core)
    OUTPUT
        density at R, phi, z
    """  

    hr = 1/2.2
    hz = 1/0.8
    
    n = params[0]
    bn = params[1]
    reff = params[2]
    fdisc = params[3]
    
    r_all = np.sqrt(R**2. + z**2.)
    r_sun = np.sqrt(_R0**2. + _z0**2.)  
    
    dens = np.exp(-bn * ((r_all/reff)**(1./n) - 1.))
    sundens = np.exp(-bn * ((r_sun/reff)**(1./n) - 1.))
        
    # disc density (the normalisation constant that measures the mass is removed when normalising by Sun density)
    diskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(z))
    diskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(_z0))
    
    # split or not the total density into halo density and disc density
    if split:
        dens, diskdens = (1-fdisc)*(dens/sundens), (fdisc)*(diskdens/diskdens_sun)
        return dens, diskdens
    else:
        dens = (1-fdisc)*(dens/sundens)+(fdisc*diskdens/diskdens_sun)
        return dens
    
# Einasto
       
def einasto(R, phi, z, params = [2., 5.], split=False):
    """
    Einasto and exponential disc model
    
    INPUT
    params = [n(0), reb(1)] 
   
    OUTPUT
    density at R, phi, z
    """
    
    _R0 = 8.275
    _z0 = 0.02
    
    # define the parameters
    n = params[0]
    rbreak = params[1]
  
    # halo density
    # compute dn (for n>=0.5 Graham et al 2006)
    dn = 3.*n -1./3. + 0.0079/n
    r_sun = np.sqrt(_R0**2 + _z0**2)
    r_all = np.sqrt(R**2 + z**2)
    dens = np.exp(-dn*((r_all/rbreak)**(1./n)-1))
    sundens = np.exp(-dn*((r_sun/rbreak)**(1./n)-1))
   
    dens = dens/sundens
    return dens    


def einasto_axisymmetric(R, phi, z, params = [2., 5., 1.], split=False):
    """
    Einasto and exponential disc model
    
    INPUT
    params = [n(0), reb(1), q(2)] 
   
    OUTPUT
    density at R, phi, z
    """
    
    _R0 = 8.275
    _z0 = 0.02
    
    # define the parameters
    n = params[0]
    rbreak = params[1]
  
    # halo density
    # compute dn (for n>=0.5 Graham et al 2006)
    dn = 3.*n -1./3. + 0.0079/n
    r_sun = np.sqrt(_R0**2 + _z0**2/params[2]**2)
    r_all = np.sqrt(R**2 + z**2/params[2]**2)
    dens = np.exp(-dn*((r_all/rbreak)**(1./n)-1))
    sundens = np.exp(-dn*((r_sun/rbreak)**(1./n)-1))
   
    dens = dens/sundens
    return dens    

def einasto_triaxial(R, phi, z, params = [2., 5., 1., 1.], split=False):
    """
    Einasto and exponential disc model
    
    INPUT
    params = [n(0), reb(1), q(2), p(3)] 
   
    OUTPUT
    density at R, phi, z
    """
    
    _R0 = 8.275
    _z0 = 0.02
    
    # define the parameters
    n = params[0]
    rbreak = params[1]
    p = params[2]
    q = params[3]
    
    # Convert from cylindrical coords to cartesian
    x,y,z = R*np.cos(phi), R*np.sin(phi), z
    x_sun, y_sun, z_sun = _R0, 0., _z0    
    
    # Define radius with transformed coordinate reference frame
    r_all = np.sqrt(x**2+(y**2/p**2)+(z**2/q**2))
    r_sun = np.sqrt(x_sun**2+(y_sun**2/p**2)+(z_sun**2/q**2))
  
    # compute dn (for n>=0.5 Graham et al 2006)
    
    dn = 3.*n -1./3. + 0.0079/n
    dens = np.exp(-dn*((r_all/rbreak)**(1./n)-1))
    sundens = np.exp(-dn*((r_sun/rbreak)**(1./n)-1))
   
    dens = dens/sundens
    return dens    


def einasto_expdisc(R, phi, z, params = [2., 5., 0.], split=False):
    """
    Einasto and exponential disc model
    
    INPUT
    params = [n(0), reb(1), fdisc(2)] 
   
    OUTPUT
    density at R, phi, z
    """
    
    _R0 = 8.275
    _z0 = 0.02
    hr = 1/2.2
    hz = 1/0.8
    
    # define the parameters
    n = params[0]
    reb = params[1]
    fdisc = params[2]
  
    # halo density
    # compute dn (for n>=0.5 Graham et al 2006)
    dn = 3.*n -1./3. + 0.0079/n
    r_sun = np.sqrt(_R0**2 + _z0**2)
    r_all = np.sqrt(R**2 + z**2)
    dens = np.exp(-dn*((r_all/reb)**(1./n)-1))
    sundens = np.exp(-dn*((r_sun/reb)**(1./n)-1))
    
    # disc density (the normalisation constant that measures the mass is removed when normalising by Sun density)
    diskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(z))
    diskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(_z0))
    
    
    # split or not the total density into halo density and disc density
    if split:
        dens, diskdens = (1-fdisc)*(dens/sundens), (fdisc)*(diskdens/diskdens_sun)
        return dens, diskdens
    else:
        dens = (1-fdisc)*(dens/sundens)+(fdisc*diskdens/diskdens_sun)
        return dens
    
def einasto_expdisc_axi(R, phi, z, params = [2., 5., 1., 0.], split=False):
    """
    Einasto and exponential disc model
    
    INPUT
    params = [n(0), reb(1), q(2), fdisc(3)] 
   
    OUTPUT
    density at R, phi, z
    """
    
    _R0 = 8.275
    _z0 = 0.02
    hr = 1/2.2
    hz = 1/0.8
    
    # define the parameters
    n = params[0]
    reb = params[1]
    q = params[2]
    fdisc = params[3]
  
    # halo density
    # compute dn (for n>=0.5 Graham et al 2006)
    dn = 3.*n -1./3. + 0.0079/n
    r_sun = np.sqrt(_R0**2 + _z0**2/q**2)
    r_all = np.sqrt(R**2 + z**2/q**2)
    dens = np.exp(-dn*((r_all/reb)**(1./n)-1))
    sundens = np.exp(-dn*((r_sun/reb)**(1./n)-1))
    
    # disc density (the normalisation constant that measures the mass is removed when normalising by Sun density)
    diskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(z))
    diskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(_z0))
    
    
    # split or not the total density into halo density and disc density
    if split:
        dens, diskdens = (1-fdisc)*(dens/sundens), (fdisc)*(diskdens/diskdens_sun)
        return dens, diskdens
    else:
        dens = (1-fdisc)*(dens/sundens)+(fdisc*diskdens/diskdens_sun)
        return dens
    
    
def einasto_expdisc_tri(R, phi, z, params = [2., 5., 1., 1., 0.], split=False):
    """
    Einasto and exponential disc model
    
    INPUT
    params = [n(0), reb(1), q(2), p(3), fdisc(4)] 
   
    OUTPUT
    density at R, phi, z
    """
    
    _R0 = 8.275
    _z0 = 0.02
    hr = 1/1.8
    hz = 1/0.8
    
    # define the parameters
    n = params[0]
    reb = params[1]
    q = params[2]
    p = params[3]
    fdisc = params[4]
  
    # halo density
    # compute dn (for n>=0.5 Graham et al 2006)
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    
    dn = 3.*n -1./3. + 0.0079/n
    r_sun = np.sqrt(_R0**2 + _z0**2/q**2)
    r_all = np.sqrt(x**2 + y**2/p**2 + z**2/q**2)
    dens = np.exp(-dn*((r_all/reb)**(1./n)-1))
    sundens = np.exp(-dn*((r_sun/reb)**(1./n)-1))
    
    # disc density (the normalisation constant that measures the mass is removed when normalising by Sun density)
    diskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(z))
    diskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(_z0))
    
    
    # split or not the total density into halo density and disc density
    if split:
        dens, diskdens = (1-fdisc)*(dens/sundens), (fdisc)*(diskdens/diskdens_sun)
        return dens, diskdens
    else:
        dens = (1-fdisc)*(dens/sundens)+(fdisc*diskdens/diskdens_sun)
        return dens
    

    
def einasto_geshalo(R, phi, z, params = [2., 5., 0.1], split=False):
    
    """
    Einasto profile with a contamination parameter to assess how much GES is present in the sample
    """
   
    _R0 = 8.275
    _z0 = 0.02  
    
    # define the parameters
    n = params[0]
    rbreak = params[1]
    fhalo = params[2]
    
    # GES halo density
    # define the best fit parameters for the GES model taken from Ted's paper
    alpha = 3.49
    beta = 0.04
    p = 0.73
    q = 0.56
    eta = 0.08*2*np.pi
    theta = 1.*2*np.pi
    phi_model = 0.04
    fdisc = 0.08
    
    original_z = np.copy(z)
    tz = (phi_model*2)-1
    zvec = np.array([np.sqrt(1-tz**2)*np.cos(theta), np.sqrt(1-tz**2)*np.sin(theta), tz])
    pa = (eta*np.pi) 
    
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_zvecpa(np.dstack([x,y,z])[0], zvec, pa)
    xsun,ysun,zsun = transform_zvecpa([_R0,0.,_z0],zvec, pa)
    r_e = np.sqrt(x**2+y**2/p**2+z**2/q**2)
    r_e_sun = np.sqrt(xsun**2+ysun**2/p**2+zsun**2/q**2)
    halodens = (r_e)**(-alpha)*np.exp(-beta*r_e)
    halosundens = (r_e_sun)**(-alpha)*np.exp(-beta*r_e_sun)
    
    # define the disc density
    hr = 1/2.2
    hz = 1/0.8
    halodiskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(original_z))
    halodiskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(_z0))

    totalhalodens = (1-fdisc)*halodens/halosundens+(fdisc*halodiskdens/halodiskdens_sun)
    
    # Einasto density
    # compute dn (for n>=0.5 Graham et al 2006)
    dn = 3.*n -1./3. + 0.0079/n
    r_sun = np.sqrt(_R0**2 + _z0**2)
    r_all = np.sqrt(R**2 + z**2)
    dens = np.exp(-dn*((r_all/rbreak)**(1./n)-1))
    sundens = np.exp(-dn*((r_sun/rbreak)**(1./n)-1))
       
        
    # split or not the total density into halo density and disc density
    if split:
        dens, halodens = (1-fhalo)*(dens/sundens), (fhalo*totalhalodens)
        return dens, halodens
    else:
        dens = (1-fhalo)*(dens/sundens)+(fhalo*totalhalodens)        
        return dens
    
def einasto_cutoff(R, phi, z, params = [2., 5., 1.], split=False):
    """
    Einasto and exponential disc model
    
    INPUT
    params = [n(0), reb(1), cutoff(2)] 
   
    OUTPUT
    density at R, phi, z
    """
    
    _R0 = 8.275
    _z0 = 0.02
    hr = 1/1.8
    hz = 1/0.8
    
    # define the parameters
    n = params[0]
    rbreak = params[1]
    cutoff = params[2]
  
    # halo density
    # compute dn (for n>=0.5 Graham et al 2006)
    dn = 3.*n -1./3. + 0.0079/n
    r_sun = np.sqrt(_R0**2 + _z0**2)
    r_all = np.sqrt(R**2 + z**2)
    dens = np.exp(-dn*((r_all/rbreak)**(1./n)-1)) * np.exp(-params[2]*r_all)
    sundens = np.exp(-dn*((r_sun/rbreak)**(1./n)-1)) * np.exp(-params[2]*r_sun)  
    
    dens = dens/sundens
    return dens

    
def einasto_expdisc_cutoff(R, phi, z, params = [2., 5., 1., 0.], split=False):
    """
    Einasto and exponential disc model
    
    INPUT
    params = [n(0), reb(1), cutoff(2), fdisc(3)] 
   
    OUTPUT
    density at R, phi, z
    """
    
    _R0 = 8.275
    _z0 = 0.02
    hr = 1/1.8
    hz = 1/0.8
    
    # define the parameters
    n = params[0]
    rbreak = params[1]
    cutoff = params[2]
    fdisc = params[3]
  
    # halo density
    # compute dn (for n>=0.5 Graham et al 2006)
    dn = 3.*n -1./3. + 0.0079/n
    r_sun = np.sqrt(_R0**2 + _z0**2)
    r_all = np.sqrt(R**2 + z**2)
    dens = np.exp(-dn*((r_all/rbreak)**(1./n)-1)) * np.exp(-params[2]*r_all)
    sundens = np.exp(-dn*((r_sun/rbreak)**(1./n)-1)) * np.exp(-params[2]*r_sun)  
   
    # disc density (the normalisation constant that measures the mass is removed when normalising by Sun density)
    diskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(z))
    diskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(_z0))
    
    # split or not the total density into halo density and disc density
    if split:
        dens, diskdens = (1-fdisc)*(dens/sundens), (fdisc)*(diskdens/diskdens_sun)
        return dens, diskdens
    else:
        dens = (1-fdisc)*(dens/sundens)+(fdisc*diskdens/diskdens_sun)
        return dens
    
    
def cored_powerlaw(R, phi, z, params = [2.,5.]):
    """
    Cored power law (spherical)
    
    INPUT
    params = [alpha (0), reb(1)] 
   
    OUTPUT
    density at R, phi, z
    """
    
    _R0 = 8.275
    _z0 = 0.02
    
    r_all = np.sqrt(R**2 + z**2)
    r_sun = np.sqrt(_R0**2 + _z0**2)

    dens = (1 + r_all/params[1])**-params[0]
    sun_dens = (1 + r_sun/params[1])**-params[0]
    
    return dens/sun_dens

def cored_powerlaw_axi(R, phi, z, params = [2.,5.,0.,0.]):
    """
    Cored power law (spherical)
    
    INPUT
    params = [alpha (0), reb(1)] 
   
    OUTPUT
    density at R, phi, z
    """
    
    _R0 = 8.275
    _z0 = 0.02
    
    p = params[1]
    q = params[2]
    
    # Convert from cylindrical coords to cartesian
    x,y,z = R*np.cos(phi), R*np.sin(phi), z
    x_sun, y_sun, z_sun = _R0, 0., _z0    
    
    # Define radius with transformed coordinate reference frame
    r_all = np.sqrt(x**2+(y**2/p**2)+(z**2/q**2))
    r_sun = np.sqrt(x_sun**2+(y_sun**2/p**2)+(z_sun**2/q**2))

    dens = (1 + r_all/params[1])**-params[0]
    sun_dens = (1 + r_sun/params[1])**-params[0]
    
    return dens/sun_dens


def cored_powerlaw_plusexpdisc(R, phi, z, params = [2., 5., 1.], split=False):
    """
    Cored power law (spherical)
    
    INPUT
    params = [alpha (0), reb(1), fdisc(2)] 
   
    OUTPUT
    density at R, phi, z
    """
    
    _R0 = 8.275
    _z0 = 0.02
    hr = 1/2.2
    hz = 1/0.8
    fdisc = params[2]
    
    r_all = np.sqrt(R**2 + z**2)
    r_sun = np.sqrt(_R0**2 + _z0**2)

    dens = (1 + r_all/params[1])**-params[0]
    sun_dens = (1 + r_sun/params[1])**-params[0]
    
    # disc density (the normalisation constant that measures the mass is removed when normalising by Sun density)
    diskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(z))
    diskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(_z0))
    
    # split or not the total density into halo density and disc density
    if split:
        dens, diskdens = (1-fdisc)*(dens/sun_dens), (fdisc)*(diskdens/diskdens_sun)
        return dens, diskdens
    else:
        dens = (1-fdisc)*(dens/sun_dens)+(fdisc*diskdens/diskdens_sun)
        return dens

# def einasto_rot(R,phi,z,params=[2.,10.,0.5,0.5,0.,0.,0.],split=False):
#     """
#     Einasto model with flattening along y and z directions and yaw,pitch,roll angles. 
#     params = [n(0),rbreak(1),p(2),q(3),gamma(4),beta(5),delta(6)] 
#     """
#     _R0 = 8.178
#     _z0 = 0.02
    
#     # define the parameters
#     n = params[0]
#     rbreak = params[1]
#     p = params[2]
#     q = params[3]
#     gamma = params[4]
#     beta = params[5]
#     delta = params[6]
    
#     # Unpack Rphiz
# #     R = Rphiz[:,0]
# #     phi = Rphiz[:,1]
# #     z = Rphiz[:,2]
        
#     # Convert from cylindrical coords to cartesian
#     x,y,z = R*np.cos(phi), R*np.sin(phi), z
#     x_sun, y_sun, z_sun = _R0, 0., _z0
    
#     # Convert to rotated reference frame
#     x_c,y_c,z_c = transform_abg(np.dstack([x,y,z])[0],delta,beta,gamma)
#     x_c_sun,y_c_sun,z_c_sun = transform_abg(np.dstack([x_sun,y_sun,z_sun])[0],delta,beta,gamma)
    
#     # Define radius with transformed coordinate reference frame
#     r_e = np.sqrt(x_c**2+(y_c**2/p**2)+(z_c**2/q**2))
#     r_e_sun = np.sqrt(x_c_sun**2+(y_c_sun**2/p**2)+(z_c_sun**2/q**2))
   

#     # compute dn (for n>=0.5 Graham et al 2006)
#     dn = 3.*n -1./3. + 0.0079/n
#     # Compute the density of a broken power law
#     dens = np.zeros(len(r_e))
#     dens = np.exp(-dn*((r_e/rbreak)**(1./n)-1))
    
#     # for density at the radius of the Sun     
#     sundens = np.exp(-dn*((r_e_sun/rbreak)**(1./n)-1))

#     return dens/sundens




def triple_pw(R,phi,z,params=[2.,3.,4.,5.,10.,0.5,0.5,0.,0.,0.,0.1],split=False):
    """
    Broken power law model with flattening along y and z directions and yaw,pitch,roll angles. Also has fdisc
    contribution parameter
    params = [alpha_in(0),alpha_mid(1),alpha_out(2),rbreak1(3),rbreak2(4),p(5),q(6),gamma(7),beta(8),delta(9),fdisc(10)] 
    """
    _R0 = 8.178
    _z0 = 0.02
    
    # define the parameters
    alpha_in = params[0]
    alpha_mid = params[1]
    alpha_out = params[2]
    rbreak1 = params[3]
    rbreak2 = params[4]
    p = params[5]
    q = params[6]
    gamma = params[7]
    beta = params[8]
    delta = params[9]
    fdisc = params[10]
#     cutoff = params[10] 
    
    # Unpack Rphiz
#     R = Rphiz[:,0]
#     phi = Rphiz[:,1]
#     z = Rphiz[:,2]
        
    # Convert from cylindrical coords to cartesian
    x,y,z = R*np.cos(phi), R*np.sin(phi), z
    x_sun, y_sun, z_sun = _R0, 0., _z0
    
    # Convert to rotated reference frame
    x_c,y_c,z_c = transform_abg(np.dstack([x,y,z])[0],delta,beta,gamma)
    x_c_sun,y_c_sun,z_c_sun = transform_abg(np.dstack([x_sun,y_sun,z_sun])[0],delta,beta,gamma)
    
    # Define radius with transformed coordinate reference frame
    r_e = np.sqrt(x_c**2+(y_c**2/p**2)+(z_c**2/q**2))
    r_e_sun = np.sqrt(x_c_sun**2+(y_c_sun**2/p**2)+(z_c_sun**2/q**2))
    # define the scale heights and lenghts of the high-alpha disc from Mackereth 2017
    hr = 1/2.
    hz = 1/0.8
    # Define the density of the exponential disc
    diskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(z))
    diskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(_z0))

    # Compute the density of a broken power law
    dens = np.zeros(len(r_e))
    dens[r_e <= rbreak1] = r_e[r_e <= rbreak1]**(-alpha_in)
    dens[(r_e <= rbreak2)&(r_e>rbreak1)] = r_e[(r_e <= rbreak2)&(r_e>rbreak1)]**(-alpha_mid)
    dens[r_e > rbreak2] = r_e[r_e > rbreak2]**(-alpha_out)
    
    # for density at the radius of the Sun     
    if r_e_sun <= rbreak1:
        sundens = (r_e_sun)**(-alpha_in)
    elif (r_e_sun <= rbreak2)&(r_e_sun > rbreak1):
        sundens = (r_e_sun)**(-alpha_mid)
    elif r_e_sun > rbreak2:
        sundens = (r_e_sun)**(-alpha_out)

    # split or not the total density into halo density and disc density
    if split:
        dens, diskdens = (1-fdisc)*(dens/sundens), (fdisc)*(diskdens/diskdens_sun)
        return dens, diskdens
    else:
        dens = (1-fdisc)*(dens/sundens)+(fdisc*diskdens/diskdens_sun)
        return dens
    


def spl_pw(R,phi,z,params=[2.,0.5,0.5,0.,0.,0.,0.1],split=False):
    """
    Single power law model with flattening along y and z directions and yaw,pitch,roll angles. Also has fdisc
    contribution parameter
    params = [alpha(0),p(1),q(2),gamma(3),beta(4),delta(5),fdisc(6)] 
    """
    _R0 = 8.178
    _z0 = 0.02
    
    # define the parameters
    alpha = params[0]
    p = params[1]
    q = params[2]
    gamma = params[3]
    beta = params[4]
    delta = params[5]
    fdisc = params[6]
    
    # Unpack Rphiz
#     R = Rphiz[:,0]
#     phi = Rphiz[:,1]
#     z = Rphiz[:,2]
        
    # Convert from cylindrical coords to cartesian
    x,y,z = R*np.cos(phi), R*np.sin(phi), z
    x_sun, y_sun, z_sun = _R0, 0., _z0
    
    # Convert to rotated reference frame
    x_c,y_c,z_c = transform_abg(np.dstack([x,y,z])[0],delta,beta,gamma)
    x_c_sun,y_c_sun,z_c_sun = transform_abg(np.dstack([x_sun,y_sun,z_sun])[0],delta,beta,gamma)
    
    # Define radius with transformed coordinate reference frame
    r_e = np.sqrt(x_c**2+(y_c**2/p**2)+(z_c**2/q**2))
    r_e_sun = np.sqrt(x_c_sun**2+(y_c_sun**2/p**2)+(z_c_sun**2/q**2))
    # define the scale heights and lenghts of the high-alpha disc from Mackereth 2017
    hr = 1/2.
    hz = 1/0.8
    # Define the density of the exponential disc
    diskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(z))
    diskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(_z0))

    # Compute the density of a broken power law
    dens = np.zeros(len(r_e))
    dens = r_e**(-alpha)
    
    # for density at the radius of the Sun     
    sundens = (r_e_sun)**(-alpha)

    # split or not the total density into halo density and disc density
    if split:
        dens, diskdens = (1-fdisc)*(dens/sundens), (fdisc)*(diskdens/diskdens_sun)
        return dens, diskdens
    else:
        dens = (1-fdisc)*(dens/sundens)+(fdisc*diskdens/diskdens_sun)
        return dens
    
def spl_pw_cutoff(R,phi,z,params=[2.,0.5,0.5,0.,0.,0.,0.1,0.4],split=False):
    """
    Single power law model with flattening along y and z directions and yaw,pitch,roll angles. Also has fdisc
    contribution parameter
    params = [alpha(0),p(1),q(2),gamma(3),beta(4),delta(5),fdisc(6),cutoff(7)] 
    """
    _R0 = 8.178
    _z0 = 0.02
    
    # define the parameters
    alpha = params[0]
    p = params[1]
    q = params[2]
    gamma = params[3]
    beta = params[4]
    delta = params[5]
    fdisc = params[6]
    cutoff = params[7]
    
    # Unpack Rphiz
#     R = Rphiz[:,0]
#     phi = Rphiz[:,1]
#     z = Rphiz[:,2]
        
    # Convert from cylindrical coords to cartesian
    x,y,z = R*np.cos(phi), R*np.sin(phi), z
    x_sun, y_sun, z_sun = _R0, 0., _z0
    
    # Convert to rotated reference frame
    x_c,y_c,z_c = transform_abg(np.dstack([x,y,z])[0],delta,beta,gamma)
    x_c_sun,y_c_sun,z_c_sun = transform_abg(np.dstack([x_sun,y_sun,z_sun])[0],delta,beta,gamma)
    
    # Define radius with transformed coordinate reference frame
    r_e = np.sqrt(x_c**2+(y_c**2/p**2)+(z_c**2/q**2))
    r_e_sun = np.sqrt(x_c_sun**2+(y_c_sun**2/p**2)+(z_c_sun**2/q**2))
    # define the scale heights and lenghts of the high-alpha disc from Mackereth 2017
    hr = 1/2.
    hz = 1/0.8
    # Define the density of the exponential disc
    diskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(z))
    diskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(_z0))

    # Compute the density of a broken power law
    dens = np.zeros(len(r_e))
    dens = r_e**(-alpha)*(np.exp(-cutoff*r_e))
    
    # for density at the radius of the Sun     
    sundens = (r_e_sun)**(-alpha)

    # split or not the total density into halo density and disc density
    if split:
        dens, diskdens = (1-fdisc)*(dens/sundens), (fdisc)*(diskdens/diskdens_sun)
        return dens, diskdens
    else:
        dens = (1-fdisc)*(dens/sundens)+(fdisc*diskdens/diskdens_sun)
        return dens
    

def broken_pw(R,phi,z,params=[2.,3.,5.,0.5,0.5,0.,0.,0.,0.1],split=False):
    """
    Broken power law model with flattening along y and z directions and yaw,pitch,roll angles. Also has fdisc
    contribution parameter
    params = [alpha_in(0),alpha_out(1),rbreak(2),p(3),q(4),gamma(5),beta(6),delta(7),fdisc(8)] 
    """
    _R0 = 8.178
    _z0 = 0.02
    
    # define the parameters
    alpha_in = params[0]
    alpha_out = params[1]
    rbreak = params[2]
    p = params[3]
    q = params[4]
    gamma = params[5]
    beta = params[6]
    delta = params[7]
    fdisc = params[8]
#     cutoff = params[8]
    
    # Unpack Rphiz
#     R = Rphiz[:,0]
#     phi = Rphiz[:,1]
#     z = Rphiz[:,2]
        
    # Convert from cylindrical coords to cartesian
    x,y,z = R*np.cos(phi), R*np.sin(phi), z
    x_sun, y_sun, z_sun = _R0, 0., _z0
    
    # Convert to rotated reference frame
    x_c,y_c,z_c = transform_abg(np.dstack([x,y,z])[0],delta,beta,gamma)
    x_c_sun,y_c_sun,z_c_sun = transform_abg(np.dstack([x_sun,y_sun,z_sun])[0],delta,beta,gamma)
    
    # Define radius with transformed coordinate reference frame
    r_e = np.sqrt(x_c**2+(y_c**2/p**2)+(z_c**2/q**2))
    r_e_sun = np.sqrt(x_c_sun**2+(y_c_sun**2/p**2)+(z_c_sun**2/q**2))
    # define the scale heights and lenghts of the high-alpha disc from Mackereth 2017
    hr = 1/2.
    hz = 1/0.8
    # Define the density of the exponential disc
    diskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(z))
    diskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(_z0))

    # Compute the density of a broken power law
    dens = np.zeros(len(r_e))
    dens[r_e <= rbreak] = r_e[r_e < rbreak]**(-alpha_in)
    dens[r_e > rbreak] = r_e[r_e > rbreak]**(-alpha_out)
    
    # for density at the radius of the Sun     
    if r_e_sun <= rbreak:
        sundens = (r_e_sun)**(-alpha_in)
    elif r_e_sun > rbreak:
        sundens = (r_e_sun)**(-alpha_out)

    # split or not the total density into halo density and disc density
    if split:
        dens, diskdens = (1-fdisc)*(dens/sundens), (fdisc)*(diskdens/diskdens_sun)
        return dens, diskdens
    else:
        dens = (1-fdisc)*(dens/sundens)+(fdisc*diskdens/diskdens_sun)
        return dens

    
def broken_pw_cutoff(R,phi,z,params=[2.,3.,5.,0.5,0.5,0.,0.,0.,0.1,0.1],split=False):
    """
    Broken power law model with flattening along y and z directions and yaw,pitch,roll angles. Also has fdisc
    contribution parameter
    params = [alpha_in(0),alpha_out(1),rbreak(2),p(3),q(4),gamma(5),beta(6),delta(7),fdisc(8),cutoff(9)] 
    """
    _R0 = 8.178
    _z0 = 0.02
    
    # define the parameters
    alpha_in = params[0]
    alpha_out = params[1]
    rbreak = params[2]
    p = params[3]
    q = params[4]
    gamma = params[5]
    beta = params[6]
    delta = params[7]
    fdisc = params[8]
    cutoff = params[9]
    
    # Unpack Rphiz
#     R = Rphiz[:,0]
#     phi = Rphiz[:,1]
#     z = Rphiz[:,2]
        
    # Convert from cylindrical coords to cartesian
    x,y,z = R*np.cos(phi), R*np.sin(phi), z
    x_sun, y_sun, z_sun = _R0, 0., _z0
    
    # Convert to rotated reference frame
    x_c,y_c,z_c = transform_abg(np.dstack([x,y,z])[0],delta,beta,gamma)
    x_c_sun,y_c_sun,z_c_sun = transform_abg(np.dstack([x_sun,y_sun,z_sun])[0],delta,beta,gamma)
    
    # Define radius with transformed coordinate reference frame
    r_e = np.sqrt(x_c**2+(y_c**2/p**2)+(z_c**2/q**2))
    r_e_sun = np.sqrt(x_c_sun**2+(y_c_sun**2/p**2)+(z_c_sun**2/q**2))
    # define the scale heights and lenghts of the high-alpha disc from Mackereth 2017
    hr = 1/2.2
    hz = 1/0.8
    # Define the density of the exponential disc
    diskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(z))
    diskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(_z0))

    # Compute the density of a broken power law
    dens = np.zeros(len(r_e))
    dens[r_e <= rbreak] = r_e[r_e < rbreak]**(-alpha_in)
    dens[r_e > rbreak] = r_e[r_e > rbreak]**(-alpha_out)*(np.exp(-cutoff*r_e[r_e > rbreak]))
    
    # for density at the radius of the Sun     
    if r_e_sun <= rbreak:
        sundens = (r_e_sun)**(-alpha_in)
    elif r_e_sun > rbreak:
        sundens = (r_e_sun)**(-alpha_out)

    # split or not the total density into halo density and disc density
    if split:
        dens, diskdens = (1-fdisc)*(dens/sundens), (fdisc)*(diskdens/diskdens_sun)
        return dens, diskdens
    else:
        dens = (1-fdisc)*(dens/sundens)+(fdisc*diskdens/diskdens_sun)
        return dens

    
def broken_pw_nodisc(R,phi,z,params=[2.,3.,5.,0.5,0.5,0.,0.,0.],split=False):
    """
    Broken power law model with flattening along y and z directions and yaw,pitch,roll angles. Also has fdisc
    contribution parameter
    params = [alpha_in(0),alpha_out(1),rbreak(2),p(3),q(4),gamma(5),beta(6),delta(7)] 
    """
    _R0 = 8.178
    _z0 = 0.02
    
    # define the parameters
    alpha_in = params[0]
    alpha_out = params[1]
    rbreak = params[2]
    p = params[3]
    q = params[4]
    gamma = params[5]
    beta = params[6]
    delta = params[7]
#     cutoff = params[8]
    
    # Unpack Rphiz
#     R = Rphiz[:,0]
#     phi = Rphiz[:,1]
#     z = Rphiz[:,2]
        
    # Convert from cylindrical coords to cartesian
    x,y,z = R*np.cos(phi), R*np.sin(phi), z
    x_sun, y_sun, z_sun = _R0, 0., _z0
    
    # Convert to rotated reference frame
    x_c,y_c,z_c = transform_abg(np.dstack([x,y,z])[0],delta,beta,gamma)
    x_c_sun,y_c_sun,z_c_sun = transform_abg(np.dstack([x_sun,y_sun,z_sun])[0],delta,beta,gamma)
    
    # Define radius with transformed coordinate reference frame
    r_e = np.sqrt(x_c**2+(y_c**2/p**2)+(z_c**2/q**2))
    r_e_sun = np.sqrt(x_c_sun**2+(y_c_sun**2/p**2)+(z_c_sun**2/q**2))

    # Compute the density of a broken power law
    dens = np.zeros(len(r_e))
    dens[r_e <= rbreak] = r_e[r_e < rbreak]**(-alpha_in)
    dens[r_e > rbreak] = r_e[r_e > rbreak]**(-alpha_out)
    
    # for density at the radius of the Sun     
    if r_e_sun <= rbreak:
        sundens = (r_e_sun)**(-alpha_in)
    elif r_e_sun > rbreak:
        sundens = (r_e_sun)**(-alpha_out)

    return dens/sundens
    
    

def broken(R,phi,z,params=[2.,3.,5.,0.5,0.5]):
    """
    Broken power law model with flattening along y and z directions 
    params = [alpha_in(0),alpha_out(1),rbreak(2),p(3),q(4)] 
    """
    _R0 = 8.178
    _z0 = 0.02
    
    # define the parameters
    alpha_in = params[0]
    alpha_out = params[1]
    rbreak = params[2]
    p = params[3]
    q = params[4]
    
    # Unpack Rphiz
#     R = Rphiz[:,0]
#     phi = Rphiz[:,1]
#     z = Rphiz[:,2]
        
    # Convert from cylindrical coords to cartesian
    x,y,z = R*np.cos(phi), R*np.sin(phi), z
    x_sun, y_sun, z_sun = _R0, 0., _z0
        
    # Define radius with transformed coordinate reference frame
    r_e = np.sqrt(x**2+(y**2/p**2)+(z**2/q**2))
    r_e_sun = np.sqrt(x_sun**2+(y_sun**2/p**2)+(z_sun**2/q**2))

    # Compute the density of a broken power law
    dens = np.zeros(len(r_e))
    dens[r_e <= rbreak] = r_e[r_e <= rbreak]**(-alpha_in)
    dens[r_e > rbreak] = r_e[r_e > rbreak]**(-alpha_out)
    
    # for density at the radius of the Sun     
    if r_e_sun <= rbreak:
        sundens = (r_e_sun)**(-alpha_in)
    elif r_e_sun > rbreak:
        sundens = (r_e_sun)**(-alpha_out)

    return dens/sundens



def triaxial_single_cutoff_zvecpa_plusexpdisk(R,phi,z,params=[2.,1.,0.5,0.5,0.,0.,0.,0.01],split=False):
    """
    triaxial power law, with zvec,pa rotation and expdisk contaminant
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [alpha,beta,p,q,theta,phi,pa,fdisc]
    OUTPUT
        density at R, phi, z
    """
    original_z = np.copy(z)
    grid = False
    theta = params[4]*2*np.pi
    tz = (params[5]*2)-1
    zvec = np.array([np.sqrt(1-tz**2)*np.cos(theta), np.sqrt(1-tz**2)*np.sin(theta), tz])
    pa = (params[6]*np.pi)
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
        original_z = original_z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_zvecpa(np.dstack([x,y,z])[0], zvec,pa)
    xsun,ysun,zsun = transform_zvecpa([_R0,0.,_z0],zvec,pa)
    r_e = np.sqrt(x**2+y**2/params[2]**2+z**2/params[3]**2)
    r_e_sun = np.sqrt(xsun**2+ysun**2/params[2]**2+zsun**2/params[3]**2)
    hr = 1/2.2
    hz = 1/0.8
    diskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(original_z))
    diskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(_z0))
    dens = (r_e)**(-params[0])*np.exp(-params[1]*r_e)
    sundens = (r_e_sun)**(-params[0])*np.exp(-params[1]*r_e_sun)
    if split:
        dens, diskdens = (1-params[7])*dens/sundens, (params[7])*diskdens/diskdens_sun
        if grid:
            dens = dens.reshape(dim)
            diskdens = diskdens.reshape(dim)
        return dens, diskdens
    else:
        dens = (1-params[7])*dens/sundens+(params[7]*diskdens/diskdens_sun)
        if grid:
            dens = dens.reshape(dim)
        return dens


def triaxial_einasto(R,phi,z,params=[2.,1.,0.5,0.5,0.,0.,0.,0.01],split=False):
    """
    triaxial power law, with zvec,pa rotation and expdisk contaminant
    INPUT
        R, phi, z - Galactocentric cylindrical coordinates
        params - [alpha,beta,p,q,theta,phi,pa,feinasto]
    OUTPUT
        density at R, phi, z
    """
    original_z = np.copy(z)
    grid = False
    theta = params[4]*2*np.pi
    tz = (params[5]*2)-1
    zvec = np.array([np.sqrt(1-tz**2)*np.cos(theta), np.sqrt(1-tz**2)*np.sin(theta), tz])
    pa = (params[6]*np.pi)
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
        original_z = original_z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_zvecpa(np.dstack([x,y,z])[0], zvec,pa)
    xsun,ysun,zsun = transform_zvecpa([_R0,0.,_z0],zvec,pa)
    r_e = np.sqrt(x**2+y**2/params[2]**2+z**2/params[3]**2)
    r_e_sun = np.sqrt(xsun**2+ysun**2/params[2]**2+zsun**2/params[3]**2)

    
    
    
    diskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(original_z))
    diskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(_z0))
    dens = (r_e)**(-params[0])*np.exp(-params[1]*r_e)
    sundens = (r_e_sun)**(-params[0])*np.exp(-params[1]*r_e_sun)
    if split:
        dens, diskdens = (1-params[7])*dens/sundens, (params[7])*diskdens/diskdens_sun
        if grid:
            dens = dens.reshape(dim)
            diskdens = diskdens.reshape(dim)
        return dens, diskdens
    else:
        dens = (1-params[7])*dens/sundens+(params[7]*diskdens/diskdens_sun)
        if grid:
            dens = dens.reshape(dim)
        return dens




    