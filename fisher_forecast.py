import numpy as np
import pickle
import matplotlib.pyplot as plt
import random

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)#, encoding='latin1')

def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        f.close()


f_sky = 0.11542924245198567

#x = np.array([8.2, 13.125, 21.0,33.6,54.,86., 137.6, 220.16])
#scalecut
x = np.array([54.,86., 137.6, 220.16])
sm = x
n = len(x)

# this is for the case of 2nd and 3rd bin only
bins_for_2 = ['2_2','2_3', '3_3']#, '3_2']
bins_for_3 = ['2_2_2','3_2_2','2_3_3','3_3_3']#,'2_3_2','3_2_3']

# all bins can be considered one of these two
equibins = ['2_2', '3_3', '2_2_2', '3_3_3']
# combinatorics are different for these options
allbins  = ['2_3', '3_2_2', '2_3_3']


EE = dict()
EEE = dict()

covmats = []
labels = []

for bin in bins_for_2:

    # load everything. 3_2 is only computed as 2_3
    if bin =='3_2':
        EE[bin] = np.transpose(EE['2_3'], [0,2,1])
    else:
        temp = []
        temp.append(load_obj("./masked_noised_cov/EE_"+bin+'_0')[:,2:,2:])
        temp.append(load_obj("./masked_noised_cov/EE_"+bin+'_1')[:,2:,2:])
        EE[bin] = np.concatenate(temp, axis = 0)


for bin in bins_for_3:

    # load everything
    if bin == '2_3_2':
        EEE[bin] = np.transpose(EEE['3_2_2'], [0,2,1,3])

    elif bin == '3_2_3':
        EEE[bin] = np.transpose(EEE['2_3_3'], [0,2,1,3])
    else:
        temp = []
        temp.append(load_obj("./masked_noised_cov/EEE_"+bin+'_0')[:,2:,2:,2:])
        temp.append(load_obj("./masked_noised_cov/EEE_"+bin+'_1')[:,2:,2:,2:])
        EEE[bin] = np.concatenate(temp, axis = 0)


#  the derivatives
dkp2_dom = dict()
dkp2_ds8 = dict()

dkp3SC_dom = dict()
dkp3SC_ds8 = dict()

dkp3GM_dom = dict()
dkp3GM_ds8 = dict()

dd_all = dict()


for bin in bins_for_2:

    # these come later and are just transposes of already loaded ones
    if bin =='3_2':
        dkp2_dom[bin] = np.transpose(dkp2_dom['2_3'], [1,0])
        dkp2_ds8[bin] = np.transpose(dkp2_ds8['2_3'], [1,0])
    else:
        # finite differences for the derivatives
        dkp2_dom[bin] = (load_obj('./fisher/kp2param0_'+bin)[2:,2:] - load_obj('./fisher/kp2param2_'+bin)[2:,2:])/0.05
        dkp2_ds8[bin] = (load_obj('./fisher/kp2param0_'+bin)[2:,2:] - load_obj('./fisher/kp2param1_'+bin)[2:,2:])/0.05


for bin in bins_for_3:

    # these come later and are just transposes of already loaded ones
    if bin == '2_3_2':
        dkp3SC_dom[bin] = np.transpose(dkp3SC_dom['3_2_2'], [1,0,2])
        dkp3SC_ds8[bin] = np.transpose(dkp3SC_ds8['3_2_2'], [1,0,2])

        dkp3GM_dom[bin] = np.transpose(dkp3GM_dom['3_2_2'], [1,0,2])
        dkp3GM_ds8[bin] = np.transpose(dkp3GM_ds8['3_2_2'], [1,0,2])

        dd_all[bin] = np.transpose(dd_all['3_2_2'], [1,0,2])

    elif bin == '3_2_3':
        dkp3SC_dom[bin] = np.transpose(dkp3SC_dom['3_2_2'], [1,0,2])
        dkp3SC_ds8[bin] = np.transpose(dkp3SC_ds8['3_2_2'], [1,0,2])

        dkp3GM_dom[bin] = np.transpose(dkp3GM_dom['3_2_2'], [1,0,2])
        dkp3GM_ds8[bin] = np.transpose(dkp3GM_ds8['3_2_2'], [1,0,2])

        dd_all[bin] = np.transpose(dd_all['3_2_2'], [1,0,2])

        #finite differences
    else:
        dkp3SC_dom[bin] = (load_obj('./fisher/kp3param0SC_'+bin)[2:,2:,2:] - load_obj('./fisher/kp3param2SC_'+bin)[2:,2:,2:])/0.05/f_sky
        dkp3SC_ds8[bin] = (load_obj('./fisher/kp3param0SC_'+bin)[2:,2:,2:] - load_obj('./fisher/kp3param1SC_'+bin)[2:,2:,2:])/0.05/f_sky

        dkp3GM_dom[bin] = (load_obj('./fisher/kp3param0GM_'+bin)[2:,2:,2:] - load_obj('./fisher/kp3param2GM_'+bin)[2:,2:,2:])/0.05/f_sky
        dkp3GM_ds8[bin] = (load_obj('./fisher/kp3param0GM_'+bin)[2:,2:,2:] - load_obj('./fisher/kp3param1GM_'+bin)[2:,2:,2:])/0.05/f_sky

        dd_all[bin] = (load_obj('./fisher/kp3param0GM_'+bin)[2:,2:,2:] - load_obj('./fisher/kp3param0SC_'+bin)[2:,2:,2:])/f_sky


# method for computing cov
def covariance_jck(TOTAL_PHI,jk_r,type_cov):
      if type_cov=='jackknife':
          fact=(jk_r-1.)/(jk_r)

      elif type_cov=='bootstrap':
          fact=1./(jk_r)
      #  Covariance estimation

      average=np.zeros(TOTAL_PHI.shape[0])
      cov_jck=np.zeros((TOTAL_PHI.shape[0],TOTAL_PHI.shape[0]))
      err_jck=np.zeros(TOTAL_PHI.shape[0])


      for kk in range(jk_r):
        average+=TOTAL_PHI[:,kk]
      average=average/(jk_r)

     # print average
      for ii in range(TOTAL_PHI.shape[0]):
         for jj in range(ii+1):
              for kk in range(jk_r):
                cov_jck[jj,ii]+=TOTAL_PHI[ii,kk]*TOTAL_PHI[jj,kk]

              cov_jck[jj,ii]=(-average[ii]*average[jj]*jk_r+cov_jck[jj,ii])*fact
              cov_jck[ii,jj]=cov_jck[jj,ii]

      for ii in range(TOTAL_PHI.shape[0]):
       err_jck[ii]=np.sqrt(cov_jck[ii,ii])
     # print err_jck

      #compute correlation
      corr=np.zeros((TOTAL_PHI.shape[0],TOTAL_PHI.shape[0]))
      for i in range(TOTAL_PHI.shape[0]):
          for j in range(TOTAL_PHI.shape[0]):
            corr[i,j]=cov_jck[i,j]/(np.sqrt(cov_jck[i,i]*cov_jck[j,j]))

      average=average*fact
      return {'cov' : cov_jck,
              'err' : err_jck,
              'corr':corr,
              'mean':average}

# "key" is redshift bin choice and select3 is equi or all. Returns "foo" which is
# the index for a rank 3 tensor that gives you all the points of interest
def pick_three(key, select3):
    if select3 == 'equi':
        foo = (range(len(x)), range(len(x)), range(len(x)))

    if select3 == 'all':
        if (key in equibins):

            bar = 0
            count = int(n*(n+1)*(n+2)/6)
            ind1 = np.zeros(count, dtype = int)
            ind2 = np.zeros(count, dtype = int)
            ind3 = np.zeros(count, dtype = int)

            for i in range(n):
                for j in range(i+1):
                    for k in range(j+1):
                        ind1[bar] = i
                        ind2[bar] = j
                        ind3[bar] = k
                        bar += 1
            foo = (ind1, ind2, ind3)

        elif (key in allbins):
            ind1 = []
            ind2 = []
            ind3 = []
            for i in range(n):
                for j in range(n):
                    for k in range(j+1):
                        ind1.append(i)
                        ind2.append(j)
                        ind3.append(k)

            foo = ( np.array(ind1), np.array(ind2), np.array(ind3))
    return foo

# "key" is redshift bin choice and select2 is equi or all. Returns "foo" which is
# the index for a rank 2 tensor that gives you all the points of interest
def pick_two(key, select2):
    if select2 == 'equi':
        foo = np.diag_indices(n)

    if select2 == 'all':
        if (key in equibins):

            foo = np.tril_indices(n)

        elif (key in allbins):
            ind1 = []
            ind2 = []
            for i in range(n):
                for j in range(n):
                    ind1.append(i)
                    ind2.append(j)

            foo = ( np.array(ind1), np.array(ind2) )
    return foo


# this function computes the covariance in om-s8 space for given policy of 2nd and 3rd moment selections
# and then saves it to the covmats array.
# label is what you want this fisher forecast contour to be labeled at the end of the script. 
# random "rndm" many elements of the DV (per Mike's suggestion as a quick and dirty hopeful way of doing compression)
# Doesn't work very well unfortunately
def AddCovmat(select2 = None, select3 = None, jk_r = 200, label = '',  scheme = 'SC', rndm = None):

    #2nd moment selection
    EEflat = []
    dkp2ds8 = []
    dkp2dom = []
    #3rd moment selection
    EEEflat = []
    dkp3_ds8 = []
    dkp3_dom = []

    if (select3 is not None):

        if scheme == 'SC':
            temps8 = dkp3SC_ds8
            tempom = dkp3SC_dom
        elif scheme == 'GM':
            temps8 = dkp3GM_ds8
            tempom = dkp3GM_dom


        for key in bins_for_3:

            foo = pick_three(key, select3)

            dkp3_ds8.append(temps8[key][foo])
            dkp3_dom.append(tempom[key][foo])

        dkp3_ds8 = np.concatenate(dkp3_ds8)
        dkp3_dom = np.concatenate(dkp3_dom)

        for i in range(jk_r):
            temp = []
            for key in bins_for_3:
                foo = pick_three(key, select3)
                temp.append(EEE[key][i][foo])
            EEEflat.append(np.concatenate(temp))


        # scatter according to theory diff
        dd = []
        for key in bins_for_3:
            foo = pick_three(key, select3)
            dd.append(dd_all[key][foo])
        dd = np.concatenate(dd)
        c = np.zeros((len(dd),len(dd)))
        for h in range(len(dd)):
               c[h,h] = dd[h]**2
        u = np.random.multivariate_normal(np.zeros(len(dd)),c,jk_r)

        EEEflat = np.array(EEEflat).T + u.T



    if (select2 is not None):
        dkp2ds8 = []
        dkp2dom = []
        for key in bins_for_2:
            foo = pick_two(key, select2)
            dkp2ds8.append(dkp2_ds8[key][foo])
            dkp2dom.append(dkp2_dom[key][foo])

        dkp2ds8 = np.concatenate(dkp2ds8)
        dkp2dom = np.concatenate(dkp2dom)

        for i in range(jk_r):
            temp = []
            for key in bins_for_2:
                foo = pick_two(key, select2)
                temp.append(EE[key][i][foo])
            EEflat.append(np.concatenate(temp))

        EEflat = np.array(EEflat).T

    if (select2 is not None) and (select3 is not None):
        flat = np.concatenate((EEflat, EEEflat))
        d_xi_ds8 = np.concatenate((dkp2ds8, dkp3_ds8))
        d_xi_dom = np.concatenate((dkp2dom, dkp3_dom))
    elif (select2 is None) and (select3 is not None):
        flat = EEEflat
        d_xi_ds8 = dkp3_ds8
        d_xi_dom = dkp3_dom
    elif (select2 is not None) and (select3 is None):
        flat = EEflat
        d_xi_ds8 = dkp2ds8
        d_xi_dom = dkp2dom
    else:
        print("no policy")
        return

    if rndm is not None:
        selectrndm = random.sample(range(len(d_xi_ds8)), rndm)
        flat = flat[selectrndm,:]
        d_xi_ds8 = d_xi_ds8[selectrndm]
        d_xi_dom = d_xi_dom[selectrndm]


    Ecc = covariance_jck(flat, jk_r, 'bootstrap')

    # initialise fisher matrix
    F = np.zeros((2,2))
    # inverse measurement covariance
    P = np.linalg.inv(Ecc['cov'])

    print(P.shape)
    # derivatives of the datavector

    F[1,1] = 0.5*(np.matmul(d_xi_ds8,np.matmul(P,d_xi_ds8))+np.matmul(d_xi_ds8,np.matmul(P.T,d_xi_ds8)))
    F[0,0] = 0.5*(np.matmul(d_xi_dom,np.matmul(P,d_xi_dom))+np.matmul(d_xi_dom,np.matmul(P.T,d_xi_dom)))
    F[0,1] = F[1,0] =  0.5*(np.matmul(d_xi_dom,np.matmul(P,d_xi_ds8)) +np.matmul(d_xi_ds8,np.matmul(P,d_xi_dom)) )
    covmats.append(0.5*(np.linalg.inv(F)+np.linalg.inv(F.T)))
    labels.append(label)
    return



# only every 3rd moment covmat
# AddCovmat(select3 = 'all',  label = 'all 3rd')

#only auto 3rd
# AddCovmat(select3 = 'equi', label = 'auto 3rd')

#every 2nd moment
# AddCovmat(select2 = 'all', label = 'all 2nd')

# only auto 2rd
# AddCovmat(select2 = 'equi', label = 'auto 2nd')

# every  moment
AddCovmat(select2 = 'all', select3 = 'all', label = 'M&M')

#rndm
#AddCovmat(select2 = 'all', select3 = 'all', label = 'M&M, random', rndm = 50)

# #only equal everything
AddCovmat(select2 = 'equi', select3 = 'equi', label = 'Marco')

import pylab as mplot
mplot.rc('text', usetex=False)
mplot.rc('font', family='serif')
import getdist
from getdist.gaussian_mixtures import GaussianND
from getdist import plots, MCSamples
import copy


chains_ = []
#
for C_par in covmats:

    omem = 0.26
    sig8 = 0.84

    x = np.random.multivariate_normal(np.array([omem,sig8]),C_par,30000)
    sig8_ = np.array(x[:,1]).astype(np.float)
    om_ = np.array(x[:,0]).astype(np.float)

    ssa = np.c_[om_.T,sig8_.T]
    samples_ = MCSamples(samples=ssa,weights=np.ones(30000), names = ['Om','s8'], labels = [r'\Omega_{\rm m}',r'\sigma_8'])

    chains_.append(samples_)


g = plots.getSubplotPlotter(width_inch=7)



g.triangle_plot(chains_,['Om','s8'],#filled=[False,False,False,False,False,False,True,True,True],  contour_lws=[1.2,1.2,1.2,1.2,1.2,1.,1.],
                legend_loc='upper right',#colors=[c_2,c_23,c_3,'black','black'],contour_colors=[c_2,c_23,c_3,'black','black'],
                contour_ls =['-','-','-','-.','-'], legend_labels=labels)
