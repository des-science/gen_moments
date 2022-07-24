import os
import numpy as np
import scipy
from scipy import integrate
import matplotlib.pyplot as plt
import pickle

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)#, encoding='latin1')

def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        f.close()

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


f_sky = 0.11542924245198567


x = np.array([21.0,33.6,54.,86., 137.6, 220.16])
#scalecut
sm = x



eebin = "2_3"
eeebin = "3_2_2"

EE = []
EEE = []

for i in range(2):

    EE.append( load_obj("./masked_noised_cov/EE_" +eebin+ "_"+ str(i)))
    EEE.append(load_obj("./masked_noised_cov/EEE_"+eeebin+"_"+ str(i)))

EE_ = np.concatenate(EE, axis = 0)
EEE_ = np.concatenate(EEE, axis = 0)

EE = {'mean': np.average(EE_, axis = 0),
      'err': np.std(EE_, axis = 0)/np.sqrt(200)
    }
EEE = {'mean': np.average(EEE_, axis = 0),
      'err': np.std(EEE_, axis = 0)/np.sqrt(200)
    }

kp20 = load_obj('./fisher/kp2param0_'+eebin)
kp3sc = load_obj('./fisher/kp3param0SC_'+eeebin)/f_sky
kp3gm = load_obj('./fisher/kp3param0GM_'+ eeebin)/f_sky

i,j = np.diag_indices(len(x), 2)
i = i[:-1]
j = j[1:]

plt.errorbar(x[i], x[i]*EE['mean'][i,j], yerr= x[i]*EE['err'][i,j], label = 'simulations')
plt.plot(sm[:-1],sm[:-1]*kp20[i,j], label = 'theory, $\chi^2 =$' + str(np.sum(((EE['mean'][i,j]-kp20[i,j])/EE['err'][i,j])**2)))
plt.legend(loc = 4)
plt.xscale('log')
plt.title("2*nd moment $\\theta_2 = 1.6\\theta_1$")
plt.xlabel("$\\theta_1$ (arcmin)")
plt.ylabel("$\\theta_1 < \kappa_{\\theta_1}\kappa_{\\theta_2}>$")
plt.show()



i,j = np.diag_indices(len(x), 2)

plt.errorbar(x[i], x[i]*EE['mean'][i,j], yerr= x[i]*EE['err'][i,j], label = 'simulations')
plt.plot(sm, sm*kp20[i,j], label = 'theory, $\chi^2 =$' + str(np.sum(((EE['mean'][i,j]-kp20[i,j])/EE['err'][i,j])**2)))
plt.legend(loc = 4)
plt.xscale('log')
plt.title("2*nd moment $\\theta_2 = \\theta_1$")
plt.xlabel("$\\theta_1$ (arcmin)")
plt.ylabel("$\\theta_1 < \kappa_{\\theta_1}\kappa_{\\theta_2}>$")
plt.show()



i,j,k = np.diag_indices(len(x), 3)
i = i[:-2]
j = j[1:-1]
k = k[2:]

plt.plot(sm[:-2],sm[:-2]**2*kp3sc[i,j,k], label = 'SC, $\chi^2 =$' + str(np.sum(((EEE['mean'][i,j,k]-kp3sc[i,j,k])/EEE['err'][i,j,k])**2)))
plt.plot(sm[:-2],sm[:-2]**2*kp3gm[i,j,k], label = 'GM, $\chi^2 =$' + str(np.sum(((EEE['mean'][i,j,k]-kp3gm[i,j,k])/EEE['err'][i,j,k])**2)))
plt.errorbar(x[i], x[i]**2*EEE['mean'][i,j,k], yerr= x[i]**2*EEE['err'][i,j,k], label = 'simulations')

plt.xscale('log')
plt.legend(loc = 4)
plt.title("3rd moment $\\theta_3 = 1.6\\theta_2 = 1.6^2\\theta_1$")
plt.xlabel("$\\theta_1$ (arcmin)")
plt.ylabel("$\\theta_1^2 < \kappa_{\\theta_1}\kappa_{\\theta_2}\kappa_{\\theta_3}>$")
plt.show()

plt.plot(x[i] ,(EEE['mean'][i,j,k]-kp3sc[i,j,k])/kp3sc[i,j,k], label = 'SC, $\chi^2 =$' + str(np.sum(((EEE['mean'][i,j,k]-kp3sc[i,j,k])/EEE['err'][i,j,k])**2)))
plt.plot(x[i] ,(EEE['mean'][i,j,k]-kp3gm[i,j,k])/kp3gm[i,j,k], label = 'GM, $\chi^2 =$' + str(np.sum(((EEE['mean'][i,j,k]-kp3gm[i,j,k])/EEE['err'][i,j,k])**2)))
plt.xscale('log')
plt.legend(loc = 4)
plt.title("3rd moment $\\theta_3 = 1.6\\theta_2 = 1.6^2\\theta_1$")
plt.xlabel("$\\theta_1$ (arcmin)")
plt.ylabel("$({\\rm measurement-theory})/({\\rm theory})$")
plt.show()



i,j,k = np.diag_indices(len(x), 3)
i = i[:-1]
j = j[:-1]
k = k[1:]

plt.plot(sm[:-1],sm[:-1]**2*kp3sc[i,j,k], label = 'SC, $\chi^2 =$' + str(np.sum(((EEE['mean'][i,j,k]-kp3sc[i,j,k])/EEE['err'][i,j,k])**2)))

plt.plot(sm[:-1],sm[:-1]**2*kp3gm[i,j,k], label = 'GM, $\chi^2 =$' + str(np.sum(((EEE['mean'][i,j,k]-kp3gm[i,j,k])/EEE['err'][i,j,k])**2)))
plt.errorbar(x[i], x[i]**2*EEE['mean'][i,j,k], yerr= x[i]**2*EEE['err'][i,j,k], label = 'simulations')

plt.xscale('log')
plt.legend(loc = 4)
plt.title("3rd moment $\\theta_3 = 1.6\\theta_2 = 1.6\\theta_1$")
plt.xlabel("$\\theta_1$ (arcmin)")
plt.ylabel("$\\theta_1^2 < \kappa_{\\theta_1}\kappa_{\\theta_2}\kappa_{\\theta_3}>$")
plt.show()

plt.plot(x[i] ,(EEE['mean'][i,j,k]-kp3sc[i,j,k])/kp3sc[i,j,k], label = 'SC, $\chi^2 =$' + str(np.sum(((EEE['mean'][i,j,k]-kp3sc[i,j,k])/EEE['err'][i,j,k])**2)))
plt.plot(x[i] ,(EEE['mean'][i,j,k]-kp3gm[i,j,k])/kp3gm[i,j,k], label = 'GM, $\chi^2 =$' + str(np.sum(((EEE['mean'][i,j,k]-kp3gm[i,j,k])/EEE['err'][i,j,k])**2)))

plt.xscale('log')
plt.legend(loc = 4)
plt.title("3rd moment $\\theta_3 = 1.6\\theta_2 = 1.6\\theta_1$")
plt.xlabel("$\\theta_1$ (arcmin)")
plt.ylabel("$({\\rm measurement-theory})/({\\rm theory})$")
plt.show()



i,j,k = np.diag_indices(len(x), 3)


plt.plot(sm ,sm**2*kp3sc[i,j,k], label = 'SC, $\chi^2 =$' + str(np.sum(((EEE['mean'][i,j,k]-kp3sc[i,j,k])/EEE['err'][i,j,k])**2)))

plt.plot(sm ,sm**2*kp3gm[i,j,k], label = 'GM, $\chi^2 =$' + str(np.sum(((EEE['mean'][i,j,k]-kp3gm[i,j,k])/EEE['err'][i,j,k])**2)))
plt.errorbar(x[i], x[i]**2*EEE['mean'][i,j,k], yerr= x[i]**2*EEE['err'][i,j,k], label = 'simulations')

plt.xscale('log')
plt.legend(loc = 4)
plt.title("3rd moment $\\theta_3 = \\theta_2 = \\theta_1$")
plt.xlabel("$\\theta_1$ (arcmin)")
plt.ylabel("$\\theta_1^2 < \kappa_{\\theta_1}\kappa_{\\theta_2}\kappa_{\\theta_3}>$")
plt.show()

plt.plot(sm ,(EEE['mean'][i,j,k]-kp3sc[i,j,k])/kp3sc[i,j,k], label = 'SC, $\chi^2 =$' + str(np.sum(((EEE['mean'][i,j,k]-kp3sc[i,j,k])/EEE['err'][i,j,k])**2)))
plt.plot(sm ,(EEE['mean'][i,j,k]-kp3gm[i,j,k])/kp3gm[i,j,k], label = 'GM, $\chi^2 =$' + str(np.sum(((EEE['mean'][i,j,k]-kp3gm[i,j,k])/EEE['err'][i,j,k])**2)))

plt.xscale('log')
plt.legend(loc = 4)
plt.title("3rd moment $\\theta_3 = \\theta_2 = \\theta_1$")
plt.xlabel("$\\theta_1$ (arcmin)")
plt.ylabel("$({\\rm measurement-theory})/({\\rm theory})$")
plt.show()
