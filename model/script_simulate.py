from get_simulation import simulate_training, simulate_test
from print_figure import print_time_series

def main():
    time_series = simulate_training()
    time_series_test = simulate_test()
    print_time_series(time_series, "../figures/simulation/training_")
    print_time_series(time_series_test, "../figures/simulation/test_")

if __name__ == "__main__":
    main()
