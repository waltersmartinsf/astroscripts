# ASTROSCRIPTS

## This library is a personal library that include helpful functions for my daily work.

# import numpy as np
import glob
import time
import sys
import os

def find_files(file_string):
    return glob.glob(file_string)

def split_list(alist, wanted_parts=1):
    '''
    Function that split a array in the number of groups, call wanted_parts,
    that we want.

    ---
    INPUT:

    alist: list of elements that we want to split_list
    wanted_parts: number of smaller groups

    ---

    Example:

    A = [0,1,2,3,4,5,6,7,8]

    print len(A)/3
    print split_list(A, wanted_parts=1)
    print split_list(A, wanted_parts=2)
    print split_list(A, wanted_parts=len(A)/3)

    Output of the Example:

    3
    [[0, 1, 2, 3, 4, 5, 6, 7, 8]]
    [[0, 1, 2, 3], [4, 5, 6, 7, 8]]
    [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    ---
    Source:
    http://stackoverflow.com/questions/752308/split-list-into-smaller-lists
    '''
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
             for i in range(wanted_parts) ]


def update_progress(progress):
    """
    Progress Bar to visualize the status of a procedure
    ___
    INPUT:
    progress: percent of the data

    ___
    Example:
    print ""
    print "progress : 0->1"
    for i in range(100):
        time.sleep(0.1)
        update_progress(i/100.0)
    """
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), round(progress*100,3), status)
    sys.stdout.write(text)
    sys.stdout.flush()
