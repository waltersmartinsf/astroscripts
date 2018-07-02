# ASTROSCRIPTS

## This library is a personal library that include helpful functions for my daily work.

def find_files(file_string):
    from glob import glob
    return glob(file_string)

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