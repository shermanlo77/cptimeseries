import math

from matplotlib import pyplot as plt
import numpy as np

import data

def main():
    cardiff_coordinates = data.CITY_LOCATION["Cardiff"]
    cardiff_fine_index = data.DataDualGrid.find_nearest_latitude_longitude(
        None, cardiff_coordinates)
    cardiff_fine = [
        data.LATITUDE_ARRAY[cardiff_fine_index[0]],
        data.LONGITUDE_ARRAY[cardiff_fine_index[1]],
    ]

    cardiff_coarse_index = [
        np.argmin(np.abs(data.LATITUDE_COARSE_ARRAY - cardiff_fine[0])),
        np.argmin(np.abs(data.LONGITUDE_COARSE_ARRAY - cardiff_fine[1])),
    ]
    cardiff_coarse = [
        data.LATITUDE_COARSE_ARRAY[cardiff_coarse_index[0]],
        data.LONGITUDE_COARSE_ARRAY[cardiff_coarse_index[1]],
    ]

    cardiff_fine = np.asarray(cardiff_fine)
    cardiff_coarse = np.asarray(cardiff_coarse)

    cardiff_dist = math.sqrt(np.sum(np.square(cardiff_fine - cardiff_coarse)))
    cardiff_dist *= data.RADIUS_OF_EARTH*math.pi/180/1000

    distance_array = []
    for lat in data.LATITUDE_ARRAY:
        for long in data.LONGITUDE_ARRAY:
            fine = [lat, long]
            coarse_index = [
                np.argmin(np.abs(data.LATITUDE_COARSE_ARRAY - fine[0])),
                np.argmin(np.abs(data.LONGITUDE_COARSE_ARRAY - fine[1])),
            ]
            coarse = [
                data.LATITUDE_COARSE_ARRAY[coarse_index[0]],
                data.LONGITUDE_COARSE_ARRAY[coarse_index[1]],
            ]
            fine = np.asarray(fine)
            coarse = np.asarray(coarse)
            dist = math.sqrt(np.sum(np.square(fine - coarse)))
            distance_array.append(dist)

    distance_array = np.asarray(distance_array)
    distance_array *= data.RADIUS_OF_EARTH*math.pi/180/1000

    plt.figure()
    ax = plt.gca()
    plt.hist(distance_array)
    ax.axvline(cardiff_dist, color='k')
    plt.xlabel("distance (km)")
    plt.show()

if __name__ == "__main__":
    main()
