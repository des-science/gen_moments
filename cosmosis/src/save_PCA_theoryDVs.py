


def filter_full_data(data_full, params):
    filter_type = params['type']
    
    sm_all = data_full['sm_all']
    id_kp2_all_finite = data_full['id_kp2_all_finite']
    ind_filter_kp2 = []
    for jid in range(len(id_kp2_all_finite)):
        jz1, jz2, i, j = id_kp2_all_finite[jid]
        if filter_type == 'auto_zbin':
            if jz1 == jz2:
                ind_filter_kp2.append(jid)

        if filter_type == 'auto_smbin':
            if i == j:
                ind_filter_kp2.append(jid)

        if filter_type == 'auto_zbin_gtsc':
            sc_all = params['sc_all']
            if jz1 == jz2:
                if (sm_all[i] > sc_all[jz1]) & (sm_all[j] > sc_all[jz2]):
                    ind_filter_kp2.append(jid)

        if filter_type == 'auto_smbin_gtsc':
            sc_all = params['sc_all']            
            if i == j:
                if (sm_all[i] > sc_all[jz1]) & (sm_all[j] > sc_all[jz2]):
                    ind_filter_kp2.append(jid)

        if filter_type == 'all_gtsc':
            sc_all = params['sc_all']            
            if (sm_all[i] > sc_all[jz1]) & (sm_all[j] > sc_all[jz2]):
                ind_filter_kp2.append(jid)
    ind_filter_kp2 = np.array(ind_filter_kp2)
    data_kp2_lhs_filter = data_full['kp2_all_lhs'][:, ind_filter_kp2]

    id_kp3_all_finite = data_full['id_kp3_all_finite']
    ind_filter_kp3 = []
    for jid in range(len(id_kp2_all_finite)):
        jz1, jz2, jz3, i, j, k = id_kp3_all_finite[jid]
        if filter_type == 'auto_zbin':
            if (jz1 == jz2) and (jz2 == jz3):
                ind_filter_kp3.append(jid)

        if filter_type == 'auto_smbin':
            if (i == j) and (j == k):
                ind_filter_kp3.append(jid)

        if filter_type == 'auto_zbin_gtsc':
            sc_all = params['sc_all']
            if (jz1 == jz2) and (jz2 == jz3):
                if (sm_all[i] > sc_all[jz1]) & (sm_all[j] > sc_all[jz2]) & (sm_all[k] > sc_all[jz3]):
                    ind_filter_kp3.append(jid)

        if filter_type == 'auto_smbin_gtsc':
            sc_all = params['sc_all']            
            if (i == j) and (j == k):
                if (sm_all[i] > sc_all[jz1]) & (sm_all[j] > sc_all[jz2]) & (sm_all[k] > sc_all[jz3]):
                    ind_filter_kp3.append(jid)

        if filter_type == 'all_gtsc':
            sc_all = params['sc_all']            
            if (sm_all[i] > sc_all[jz1]) & (sm_all[j] > sc_all[jz2]) & (sm_all[k] > sc_all[jz3]):
                ind_filter_kp3.append(jid)
    ind_filter_kp3 = np.array(ind_filter_kp3)
    data_kp3_lhs_filter = data_full['kp3_all_lhs'][:, ind_filter_kp3]

    return data_kp2_lhs_filter, data_kp3_lhs_filter
    