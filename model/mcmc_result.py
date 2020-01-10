import compound_poisson as cp
import matplotlib.pyplot as plot
import numpy as np
import joblib

time_series = joblib.load("mcmc_result.zlib")

true_parameter = time_series.get_parameter_vector()
chain = np.asarray(time_series.parameter_sample)
for i in range(time_series.n_parameter):
    chain_i = chain[:,i]
    plot.figure()
    plot.plot(chain_i)
    plot.hlines(true_parameter[i], 0, len(chain)-1)
    plot.ylabel("Parameter "+str(i))
    plot.xlabel("Sample number")
    plot.savefig("parameter_sample_"+str(i)+".eps")
    plot.show()
    plot.close()

chain = []
for i in range(len(time_series.z_sample)):
    chain.append(np.mean(time_series.z_sample[i]))
plot.figure()
plot.plot(chain)
plot.ylabel("Mean of latent variables")
plot.xlabel("Sample number")
plot.savefig("z_sample.eps")
plot.show()
plot.close()

plot.figure()
plot.plot(np.asarray(time_series.accept_reg_array))
plot.ylabel("Acceptance rate of parameters")
plot.xlabel("Parameter sample number")
plot.savefig("accept_parameter.eps")
plot.show()
plot.close()

plot.figure()
plot.plot(np.asarray(time_series.accept_z_array))
plot.ylabel("Acceptance rate of latent variables")
plot.xlabel("Latent variable sample number")
plot.savefig("accept_z.eps")
plot.show()
plot.close()
