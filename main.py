import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
from predict_model import Model

def timeProcess(args):
    month_day = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    return np.cumsum(month_day)[args.month-1] + args.date - 1

def getRegion(lat, lon, height, width):

    lat = round((90 - lat)/90 * height * 0.5)
    lon = round(lon/180 * width * 0.5)

    return lat, lon

def convolution2d(image, kernel, bias):
    m, n = kernel.shape
    if (m == n):
        y, x = image.shape
        new_image = np.zeros((y,x))
        for i in range(y-m+1):
            for j in range(x-m+1):
                new_image[i+10][j+10] = np.sum(image[i:i+m, j:j+m]*kernel) + bias
    return new_image

def loadNEX(day):

    wind_ds = netCDF4.Dataset('nex_data/wind_daily_2014-2014.nc', mode='r')
    temp_ds = netCDF4.Dataset('nex_data/tas_daily_2014-2014.nc', mode='r')
    hum_ds = netCDF4.Dataset('nex_data/shum_daily_2014-2014.nc', mode='r')

    return wind_ds.variables['wind'][day], temp_ds.variables['tas'][day], hum_ds.variables['shum'][day]

def drawPlot(subtitle, data, cmap_type):
    # plt.hist2d(x, y)

    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])

    fig, ax = plt.subplots()
    map = plt.imread("input_map/map24x120_5.png")
    plt.imshow(map, extent=[0, data.shape[1], 0, data.shape[0]])

    fig.suptitle(subtitle)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.contourf(x, y, data, 8, alpha=0.4, cmap=cmap_type)
    plt.colorbar()

    plt.savefig('output_map/' + subtitle + '.png')

    # plt.show()


def main(argv=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--year', default='2014', type=int)
    parser.add_argument('--month', default='1', type=int)
    parser.add_argument('--date', default='4', type=int)
    parser.add_argument('--lat', default='24.04023', type=float)
    parser.add_argument('--lon', default='120.56805', type=float)
    args = parser.parse_args()

    day = timeProcess(args)

    # model training
    fire_model = Model()
    model = fire_model.model
    distance_kernel = fire_model.distance_kernel

    # load nex data
    wind_ds, temp_ds, hum_ds = loadNEX(day)

    # transfer specific humidity to relative humidity (roughly)
    # sh <- rh * 2.541e6 * exp(-5415.0 / T) * 18/29
    # https://github.com/PecanProject/pecan/tree/master/modules/data.atmosphere
    hum_ds = hum_ds/4093833.33/np.exp(-5415/temp_ds)*100

    # transfer K to C
    temp_ds -= 273.5

    # draw source data
    # drawPlot('Wind Speed', wind_ds, plt.cm.RdGy)
    # drawPlot('Temperature', temp_ds, plt.cm.bwr)
    # drawPlot('Humidity', hum_ds, plt.cm.Spectral)

    # get particular region data
    lat, lon = getRegion(round(args.lat/0.25)*0.25, round(args.lon/0.25)*0.25, hum_ds.shape[0], hum_ds.shape[1])

    wind_region_idx = np.array([(wind_ds[lat-1, lon-1], wind_ds[lat, lon-1], wind_ds[lat+1, lon-1])
                               , (wind_ds[lat-1, lon], wind_ds[lat, lon], wind_ds[lat+1, lon])
                               , (wind_ds[lat-1, lon+1], wind_ds[lat, lon+1], wind_ds[lat+1, lon+1])])

    temp_region_idx = np.array([(temp_ds[lat-1, lon-1], temp_ds[lat, lon-1], temp_ds[lat+1, lon-1])
                               , (temp_ds[lat-1, lon], temp_ds[lat, lon], temp_ds[lat+1, lon])
                               , (temp_ds[lat-1, lon+1], temp_ds[lat, lon+1], temp_ds[lat+1, lon+1])])

    hum_region_idx = np.array([(hum_ds[lat-1, lon-1], hum_ds[lat, lon-1], hum_ds[lat+1, lon-1])
                               , (hum_ds[lat-1, lon], hum_ds[lat, lon], hum_ds[lat+1, lon])
                               , (hum_ds[lat-1, lon+1], hum_ds[lat, lon+1], hum_ds[lat+1, lon+1])])

    temp_region = cv2.resize(temp_region_idx, (200, 200), interpolation=cv2.INTER_LINEAR)
    wind_region = cv2.resize(wind_region_idx, (200, 200), interpolation=cv2.INTER_LINEAR)
    hum_region = cv2.resize(hum_region_idx, (200, 200), interpolation=cv2.INTER_LINEAR)

    drawPlot('Wind Speed (Region)', wind_region, plt.cm.RdGy)
    drawPlot('Temperature (Region)', temp_region, plt.cm.bwr)
    drawPlot('Humidity (Region)', hum_region, plt.cm.Spectral)

    # evaluate fire index using fire model
    fire_index_map = np.zeros((200, 200))
    data = np.concatenate((temp_region/fire_model.temp_parameter, hum_region/fire_model.RH_parameter, wind_region/fire_model.wind_parameter), axis=0)
    for i in range(200):
        for j in range(200):
            fire_index_map[i][j] = model.predict(np.array([[data[i][j], data[i+200][j], data[i+400][j]]]))
    drawPlot('Wild Fire Model Index (Region)', np.power(fire_index_map/fire_index_map.max()*10, 2), plt.cm.jet)

    # mark fire site
    fire_lat = round((args.lat - (round(args.lat/0.25)*0.25-0.25))/0.5*200)
    fire_lon = round((args.lon - (round(args.lon/0.25)*0.25-0.25))/0.5*200)

    # fire simulation
    fire_source = np.zeros((200, 200))
    for i in range(11):
        for j in range(11):
            fire_source[fire_lat+i-5][fire_lon+j-5] = 1

    drawPlot('Fire Region', fire_source, plt.cm.hot)

    # get dangerous index
    fire_predict_region = np.zeros((200, 200))
    fire_predict_region = convolution2d(fire_source, distance_kernel, 0)
    drawPlot('Fire Danger Index', (fire_predict_region*fire_index_map)/(fire_predict_region*fire_index_map).max()*90, plt.cm.hot)


if __name__ == '__main__':
    main()
