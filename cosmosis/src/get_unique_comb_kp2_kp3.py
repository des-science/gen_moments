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



def save_unique_combs_general(nsm, nzbins):
    """
    This function is for general case where each tomographic bin is probing some different type of sample.
    In case of kappa that is not the case as each bin is just probing the dark matter with different weighting.
    So use the simple function above.    
    """
    z_all = []
    for jz in range(nzbins):
        z_all.append('z'+str(jz))

    s_all = []
    for js in range(nsm):
        s_all.append('s'+str(js))

    unique_combo_corr2 = []
    id_corr2_unique = []
    ntot_corr2 = 0
    for js1 in range(nsm):
        for js2 in range(nsm):
            ntot_corr2 += 1
            comb_p0 = s_all[js1] + s_all[js2]  
            comb_p1 = s_all[js2] + s_all[js1] 

            comb_all = [comb_p0, comb_p1]
            any_comb_in_arr = False
            for comb in comb_all:
                if comb in unique_combo_corr2:
                    any_comb_in_arr = True
            if not any_comb_in_arr:
                unique_combo_corr2.append(comb_p0)
                id_corr2_unique.append((js1, js2))

    unique_combo_kp2 = []
    id_kp2_unique = []
    ntot_kp2 = 0
    for jz1 in (range(nzbins)):
        for jz2 in range(nzbins):
            for js1 in range(nsm):
                for js2 in range(nsm):
                    
                    ntot_kp2 += 1
                    comb_p0 = z_all[jz1] + s_all[js1] + z_all[jz2] + s_all[js2]  
                    comb_p1 = z_all[jz2] + s_all[js2] + z_all[jz1] + s_all[js1] 

                    comb_all = [comb_p0, comb_p1]
                    any_comb_in_arr = False
                    for comb in comb_all:
                        if comb in unique_combo_kp2:
                            any_comb_in_arr = True
                    if not any_comb_in_arr:
                        unique_combo_kp2.append(comb_p0)
                        id_kp2_unique.append((jz1, jz2, js1, js2))

    
    unique_combo_corr3 = []
    id_corr3_unique = []
    ntot_corr3 = 0
    for js1 in range(nsm):
        for js2 in range(nsm):
            for js3 in range(nsm):
                ntot_corr3 += 1
                comb_p0 = s_all[js1] + s_all[js2] + s_all[js3]  
                comb_p1 = s_all[js1] + s_all[js3] + s_all[js2]
                comb_p2 = s_all[js2] + s_all[js1] + s_all[js3]  
                comb_p3 = s_all[js2] + s_all[js3] + s_all[js1]  
                comb_p4 = s_all[js3] + s_all[js1] + s_all[js2] 
                comb_p5 = s_all[js3] + s_all[js2] + s_all[js1]

                comb_all = [comb_p0, comb_p1, comb_p2, comb_p3, comb_p4, comb_p5]
                any_comb_in_arr = False
                for comb in comb_all:
                    if comb in unique_combo_corr3:
                        any_comb_in_arr = True
                if not any_comb_in_arr:
                    unique_combo_corr3.append(comb_p0)
                    id_corr3_unique.append((js1, js2, js3))
    
    unique_combo_kp3 = []
    id_kp3_unique = []
    ntot_kp3 = 0
    for jz1 in (range(nzbins)):
        for jz2 in range(nzbins):
            for jz3 in range(nzbins):
                for js1 in range(nsm):
                    for js2 in range(nsm):
                        for js3 in range(nsm):
                            ntot_kp3 += 1
                            comb_p0 = z_all[jz1] + s_all[js1] + z_all[jz2] + s_all[js2] + z_all[jz3] + s_all[js3]  
                            comb_p1 = z_all[jz1] + s_all[js1] + z_all[jz3] + s_all[js3] + z_all[jz2] + s_all[js2]
                            comb_p2 = z_all[jz2] + s_all[js2] + z_all[jz1] + s_all[js1] + z_all[jz3] + s_all[js3]  
                            comb_p3 = z_all[jz2] + s_all[js2] + z_all[jz3] + s_all[js3] + z_all[jz1] + s_all[js1]  
                            comb_p4 = z_all[jz3] + s_all[js3] + z_all[jz1] + s_all[js1] + z_all[jz2] + s_all[js2] 
                            comb_p5 = z_all[jz3] + s_all[js3] + z_all[jz2] + s_all[js2] + z_all[jz1] + s_all[js1]
                            
                            comb_all = [comb_p0, comb_p1, comb_p2, comb_p3, comb_p4, comb_p5]
                            any_comb_in_arr = False
                            for comb in comb_all:
                                if comb in unique_combo_kp3:
                                    any_comb_in_arr = True
                            if not any_comb_in_arr:
                                unique_combo_kp3.append(comb_p0)
                                id_kp3_unique.append((jz1, jz2, jz3, js1, js2, js3))

    return id_corr2_unique, id_kp2_unique, id_corr3_unique, id_kp3_unique



