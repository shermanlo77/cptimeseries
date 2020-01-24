import joblib
import matplotlib.pyplot as plot
from get_simulation import simulate_training, simulate_test
from print_figure import print_time_series, print_forecast

def main():
    
    time_series = joblib.load("results/simulate_sgd.zlib")
    print("Fitted model:",time_series)
    
    directory = "../figures/simulation/sgd/"
    
    plot.figure()
    ax = plot.gca()
    ln_l_array = time_series.ln_l_array
    ln_l_stochastic_index = time_series.ln_l_stochastic_index
    for i in range(len(ln_l_stochastic_index)-1):
        start = ln_l_stochastic_index[i]-1
        end = ln_l_stochastic_index[i+1]
        if i%2 == 0:
            linestyle = "-"
        else:
            linestyle = ":"
        ax.set_prop_cycle(None)
        plot.plot(range(start, end), ln_l_array[start:end], linestyle=linestyle)
    plot.axvline(x=time_series.ln_l_max_index, linestyle='--')
    plot.xlabel("Number of EM steps")
    plot.ylabel("log-likelihood")
    plot.savefig(directory + "fitted_ln_l.pdf")
    plot.close()
    
    time_series_true = simulate_training()
    time_series_future = simulate_test()
    print_forecast(time_series, 
                   time_series_true.y_array, 
                   time_series_future.x,
                   time_series_future.y_array,
                   directory)
    
    time_series.simulate()
    print_time_series(time_series, directory + "fitted_")

if __name__ == "__main__":
    main()
