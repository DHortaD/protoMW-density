import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mwdust
import tqdm
import dill as pickle
import multiprocessing
import utils
import densprofiles
from isodist import FEH2Z, Z2FEH
import corner
import matplotlib
import safe_colours
from galpy.util import bovy_plot #only necessary if you want to plot...
import os
import sys
import matplotlib.patheffects as PathEffects
import dill as pickle
import astropy.io.fits as fits


os.environ['RESULTS_VERS'] = 'dr17'
os.environ['SDSS_LOCAL_SAS_MIRROR'] = '/Users/qpasa/Desktop/halo-density/data/'

# https://github.com/astrojimig/apogee/tree/dr17-selection
sys.path.insert(1,'/Users/qpasa/Packages/apogee/')

import apogee.select as apsel
import apogee.tools.read as apread

columnwidth = 240./72.27
textwidth = 504.0/72.27

force = False
if os.path.exists('../sav/apogeeCombinedSF_DR17.dat') and not force:
    with open('../sav/apogeeCombinedSF_DR17.dat', 'rb') as f:
        apo = pickle.load(f)
else:
    apo = apsel.apogeeCombinedSelect(year=10)
    with open('../sav/apogeeCombinedSF_DR17.dat', 'wb') as f:
        pickle.dump(apo, f)
force = False

def add_deltaM_column(iso):
    deltaM = []
    logg_min_lim = 0.5
    logg_max_lim = 3.5
    for i in range(len(iso['MH'])):
        if (iso['MH'][i-1]==iso['MH'][i])&(iso['logAge'][i-1]==iso['logAge'][i])==True:
            #Same metallicity and age = same stellar population
            deltaM.append(iso['int_IMF'][i]-iso['int_IMF'][i-1])
        else:
            deltaM.append(iso['int_IMF'][i+1]-iso['int_IMF'][i])

    #Renormalize, so every population sums to 1
    for m in set(iso['MH']):
        for a in set(iso['logAge']):
            isom = (iso['MH']==m)&(iso['logAge']==a)&(iso['logg']>=logg_min_lim)&(iso['logg']<=logg_max_lim)
            #logg_min_lim and logg_max_lim are global variables, the logg limits for the giant branch cut
            dmtot = np.sum(np.array(deltaM)[isom])
        for i in range(len(deltaM)):
            if isom[i]==True:
                deltaM[i] = deltaM[i]/dmtot
    return np.array(deltaM)

#load the isochrone grid (make sure this is downloaded first!)
isorec = utils.generate_lowfeh_isogrid()
deltaM = add_deltaM_column(isorec)

import utils
import apogee.select as apsel

#load or evaluate the effective selection function on the required [Fe/H] grid (defined in 'bins')
def _calc_effsel_onelocation(i): 
    #computes the effective selection function along the line of sight of one APOGEE location.
    loc = apo._locations[i]
    if np.sum([np.nansum(apo._nspec_short[i]),np.nansum(apo._nspec_medium[i]),np.nansum(apo._nspec_long[i])]) < 1.:
        effsel = np.zeros(len(ds))
    elif apo.JKmin(loc) >= 0.5:
        print(apo.JKmin(loc),loc)
        effsel = apof(loc, ds, MH=H, JK0=(J-K))*apo.area(loc)
    elif apo.JKmin(loc) < 0.5:
        effsel = apof(loc, ds, MH=p3H, JK0=(p3J-p3K))*apo.area(loc)
    return effsel

#repeat the above for a single [Fe/H] selection across the whole range
force = False

# Set the parameters of the distance modulus grid
nsamples=2000
ndistmods=301
minmax_distmods=[6.,18.]
nthreads = int(multiprocessing.cpu_count()//2)
#load the isochrone grid (make sure this is downloaded first!)
isorec = utils.generate_lowfeh_isogrid()
# dustmap from mwdust
dmap = mwdust.Combined19()
#initiate eff sel instance
apof = apsel.apogeeEffectiveSelect(apo, dmap3d=dmap,weights=deltaM)
distmods = np.linspace(minmax_distmods[0], minmax_distmods[1], ndistmods)
ds = 10.**(distmods/5-2)
niso, p3niso = utils.APOGEE_iso_samples(nsamples, isorec, fehrange=[-2.5, -0.5], lowfehgrid=True)
H, J, K = np.array(niso['Hmag'].data), np.array(niso['Jmag'].data), np.array(niso['Ksmag'].data)
p3H, p3J, p3K = np.array(p3niso['Hmag'].data), np.array(p3niso['Jmag'].data), np.array(p3niso['Ksmag'].data)
distmods = np.linspace(minmax_distmods[0], minmax_distmods[1], ndistmods)
ds = 10.**(distmods/5-2)

if os.path.exists('../essf/effsel_grid_dr17.dat') and not force:
    sys.stdout.write('\r'+"loading saved effective selection function for APOGEE fields...\r")
    sys.stdout.flush()
    with open('../essf/effsel_grid_dr17.dat', 'rb') as f:
        outarea = pickle.load(f)
else:
    sys.stdout.write('\r'+"calculating effective selection function for APOGEE fields...\r")
    sys.stdout.flush()
    with multiprocessing.Pool(nthreads) as p:
        outarea = list(tqdm.tqdm(p.imap(_calc_effsel_onelocation, range(0,len(apo._locations))), total=len(apo._locations)))
    outarea = np.array(outarea)
    with open('../essf/effsel_grid_dr17.dat', 'wb') as f:
        pickle.dump(outarea, f)
