import dataset
import plot_data

def main():
    data = dataset.WalesTest()
    plot_data.plot_data(data)

if __name__ == "__main__":
    main()
