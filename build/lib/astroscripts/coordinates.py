"""
This script helps to find an astronomical object in the FoV of 
fits file, WHEN the world coordinate system are writed in the header.
"""
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import Angle
from astropy import wcs
from astropy.io import fits


def XYangle(RA,DEC, RA_unit=u.hourangle,DEC_unit=u.deg):
    RA = Angle(RA,RA_unit)
    DEC = Angle(DEC,DEC_unit)
    return RA, DEC

def fits_XY(image_path,RA,DEC):
    img, hdr = fits.getdata(image_path,header=True)
    img = img.astype(np.float64)
    w = wcs.WCS(hdr)
    RA_px, DEC_px = w.all_world2pix(RA,DEC,0)
    return RA_px, DEC_px, img, hdr

def matching_stars(coordinates,sources,delta=10,match_RA = True, match_DEC = False,xcentroid_label='xcentroid', ycentroid_label='ycentroid',show=True):

    if match_RA == True:
        select_stars = sources[(sources[xcentroid_label] < coordinates[0]+delta) & (sources[xcentroid_label] > coordinates[0]-delta)]
    if match_DEC == True:
        select_stars = select_stars[(select_stars[ycentroid_label] < coordinates[1]+delta) & (select_stars[ycentroid_label] > coordinates[1]-delta)]
    if show == True:
            print('Found ',len(select_stars),' objects.')
    if len(select_stars) == 1:
        status = 0
        return status, select_stars
    else:
        status = 1
        if show == True:
            print('Number of objects founded is more than one, Please, re-check the ')
        return status, select_stars