import sys, os
import numpy as np
import ast


def save_unique_combs(nsm, nzbins):
    id_corr2_unique = []
    for i in range(nsm):
        for j in range(nsm):
            if i <= j:
                id_corr2_unique.append((i,j))

    id_kp2_unique = []
    for jz1 in range(nzbins):
        for jz2 in range(nzbins):
            if jz1 <= jz2:
                for i in range(nsm):
                    for j in range(nsm):
                        if i <= j:
                            id_kp2_unique.append((jz1, jz2, i, j))
    
    id_corr3_unique = []
    for i in range(nsm):
        for j in range(nsm):
            for k in range(nsm):
                if (k >= j and j >= i):
                    id_corr3_unique.append((i,j,k))
    
    id_kp3_unique = []
    for jz1 in range(nzbins):
        for jz2 in range(nzbins):
            for jz3 in range(nzbins):
                if (jz3 >= jz2 and jz2 >= jz1):
                    for i in range(nsm):
                        for j in range(nsm):
                            for k in range(nsm):
                                if (k >= j and j >= i):
                                    id_kp3_unique.append((jz1, jz2, jz3, i, j, k))

    return id_corr2_unique, id_kp2_unique, id_corr3_unique, id_kp3_unique
