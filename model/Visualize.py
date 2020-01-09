import numpy as np
import numpy.random as random
import pdb
import math
import matplotlib.pyplot as plot

chain = np.load('Posterior_Sample.npz')['chain']
zchain = np.load('Posterior_Sample.npz')['zchain']
zacc = np.load('Posterior_Sample.npz')['zacc']
regacc = np.load('Posterior_Sample.npz')['regarr']
true_parameter = np.load('Posterior_Sample.npz')['true_parameter']

print(regacc.shape)


for i in range(chain.shape[1]):
    plot.figure()
    plot.plot(chain[:,i])
    plot.plot([0, chain.shape[0]],[true_parameter[i],true_parameter[i]])
    plot.show()
    plot.savefig('Param_'+str(i)+'.eps')
    plot.close()

zchain=time_series.z_sample, zacc=time_series.accept_z_array, regarr

chain = np.sum(np.asarray(),1)
plot.figure()
plot.plot(chain)
plot.show()

accept = np.asarray(time_series.accept_reg_array)
plot.figure()
plot.plot(accept)
plot.show()

accept = np.asarray(time_series.accept_z_array)
plot.figure()
plot.plot(accept)
plot.show()