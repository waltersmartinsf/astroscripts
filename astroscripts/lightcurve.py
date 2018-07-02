"""
LIGHTCURVE routines

""""

import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import scipy #pearson correlation
import matplotlib.pyplot as plt

def error_flux(hoststar_flux,hoststar_eflux,ref_star_flux,ref_star_eflux):
    """
    Obtain the errorbar normalized flux for a planetary transit normalized by a reference star in the FoV.
    """
    _flux = hoststar_flux/ref_star_flux
    part1 = (hoststar_eflux/hoststar_flux)**2
    part2 = (ref_star_eflux/ref_star_flux)**2
    _eflux = _flux*np.sqrt(np.array(part1.values + part2.values))
    return error_flux