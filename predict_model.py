import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error

class Model(object):
    
    def __init__(self):
        self.file_name = "fire_data/forestfires.csv"
        self.temp_parameter = 0
        self.RH_parameter = 0
        self.wind_parameter = 0
        self.area_parameter = 0

        weather_data = self.dataPreprocess()
        self.model = self.dataTraining(weather_data)
        self.distance_kernel = self.makeGaussian(21, 12, center=[10, 10])

    def dataPreprocess(self):
        nRowsRead = 1000
        data = pd.read_csv(self.file_name, delimiter=',', nrows=nRowsRead)
        data.dataframeName = self.file_name

        # data.month.replace(('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'),
        #                         (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), inplace=True)
        # data.day.replace(('mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'), (1, 2, 3, 4, 5, 6, 7), inplace=True)

        self.temp_parameter = data.temp.max() - data.temp.min()
        data.temp = data.temp / self.temp_parameter
        self.RH_parameter = data.RH.max() - data.RH.min()
        data.RH = data.RH / self.RH_parameter
        self.wind_parameter = data.wind.max() - data.wind.min()
        data.wind = data.wind / self.wind_parameter
        self.area_parameter = data.area.max() - data.area.min()
        data.area = data.area / self.area_parameter

        return data


    def dataTraining(self, data):
        train_x = data.values[:, 2:-1]
        train_y = data.values[:, -1]

        model = ExtraTreesRegressor()

        model.fit(train_x, train_y)

        predictions = model.predict(train_x)

        # Evaluate the model
        score = explained_variance_score(train_y, predictions)
        mae = mean_absolute_error(predictions, train_y)

        msg = "%f (%f)" % (score, mae)
        print(msg)

        return model

    def makeGaussian(self, size, fwhm=3, center=None):

        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]

        if center is None:
            x0 = y0 = size // 2
        else:
            x0 = center[0]
            y0 = center[1]

        return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)





