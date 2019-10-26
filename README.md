# NASA2019 Team ET760
Wildfire Risk Predict Model

## Mission
Use the data provided by OpenNEX to predict the regional wildfire risk. The output of this model is shown below (Higher value represents higher risk of wildfire)

![](https://i.imgur.com/sF81sZM.png)

## Training Data
1. Global Meteorological Forcing Dataset for land surface modeling-version 3-0.25 degree (2014): http://hydrology.princeton.edu/data/pgf/v3/0.25deg/daily/
2. Forest Fires Dataset released by the University of Minho (2007): 
http://www3.dsi.uminho.pt/pcortez/forestfires/
3. Fire Information for Resource Management System (FIRMS)-VIIRS-Taiwan-2014/01/01-2014/01/31
https://earthdata.nasa.gov/earth-observation-data/near-real-time/firms

## Notes
1. The dataset [1] is provided by the OpenNEX. It includes global information about humidity, temperature, and wind speed. Yet, it doesn’t provide any fire-related information.
2. The dataset [2] provides the small region wildfire records in Portuguese. It also records the weather information and the burning area size each of the fire emergency moments.
3. We assume that the burning area could be represented as the serious level of wildfire, so that, we want to find the model to find the relationship between the weather factors (humidity, temperature, and wind speed) and the serious level of wildfire.
4. We normalize all the data into 0~1. Besides, we use several different models during training, including Lasso, KNN, SVM, etc with the loss function mean absolute error. Extra-trees regressor has the best performance under this circumstance. Therefore, we take that model as our training model.
5. Finally, we use the dataset [3] and find the latitude and longitude coordinates information of a real fire case. Once the user inputs time and coordinates information, the program will show the regional weather information in ±0.25 degree. Then, we will predict the wildfire dangerous region with the distance filter (which is a Gaussian filter) and our model. The following image presents the wildfire danger index (Higher value represents higher risk of wildfire)

![](https://i.imgur.com/8pGgflo.png)


## How to Use
The command of this program is shown below, all the arguments are adjustable:

`    python main.py --year 2014 --month 1 --date 4 --lat 24.04023 --lon 120.56805`
* lat: latitude
* lon: longitude 


