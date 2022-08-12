from math import floor


def print_info(ncols, info, placeholder='='):
    info_len = len(info)
    placeholder_len = ncols - info_len
    
    print((int(placeholder_len/2)-1) * placeholder, end=' ')
    print(info, end=' ')
    print((int((placeholder_len+1)/2)-1) * placeholder)
    