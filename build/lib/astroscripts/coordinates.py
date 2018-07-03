"""
This script helps to find an astronomical object in the FoV of 
fits file, WHEN the world coordinate system are writed in the header.
"""
import numpy as np
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