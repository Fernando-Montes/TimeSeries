import pandas as pd
import numpy as np
from random import sample
import math

from statsmodels.tsa.arima_model import ARIMA

from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn import pipeline
from sklearn import linear_model
from sklearn import neighbors
from sklearn.svm import LinearSVR

def next_batch(series, t, fit_range):
    train = series[t:(t+fit_range)]
    train.index = range(0,fit_range) # To avoid issues when training ARIMA
    return train

def fitting(train, model_sel, steps_ahead):
    '''
    Returns prediction
    '''

    # Select models
    if model_sel[0] == 'ARIMA':
        model = ARIMA( train['Close'], order=model_sel[1] )
        model_fit = model.fit()
        res = model_fit.forecast(steps=steps_ahead) # Forecasting steps_ahead
        output = res[0][len(res[0])-1] # Only consider steps_ahead
    else:
        # Prepare data
        X = np.c_[train.index]
        y = np.c_[train['Close']]
        poly = preprocessing.PolynomialFeatures(degree=3, include_bias=False)
        scaler = preprocessing.StandardScaler()

        if model_sel[0] == 'poly':
            model = linear_model.Ridge(alpha=model_sel[1])
        elif model_sel[0] == 'SVM':
            model = LinearSVR(C=model_sel[1][0], epsilon=model_sel[1][1])

        modelPip = pipeline.Pipeline([('poly', poly), ('scal', scaler), ('modl', model)])
        model_fit = modelPip.fit(X, y)
        output = model_fit.predict([[X[len(X)-1][0]+steps_ahead]] )
    return output.flatten()[0]

def rollingEstimate(series, model_sel, steps_ahead, fit_range, verbose=False, fastRMSE=False, numInstances=200):
    '''
    Returns RMSE of a steps_ahead rolling estimate using fit_range
    Returns predictions
    '''

    results = series[(fit_range+steps_ahead-1):len(series)]
    results = results.assign(Pred=None)

    # Rolling forecast
    if fastRMSE:
        rangeForecast = sample( range(0, len(series)-fit_range-steps_ahead), numInstances)
    else:
        rangeForecast = range(0, len(series)-fit_range) # Includes estimates ahead
    for t in rangeForecast:
        train = next_batch(series, t, fit_range)
        output = fitting(train, model_sel, steps_ahead)
        results.loc[t+fit_range+steps_ahead-1,'Pred'] = output
        if t%200 == 0 and verbose == True:
            print('t={0} - expected={1:.2f} - predicted={2:.2f}'.format(
                fit_range+steps_ahead-1+t, results.loc[fit_range+steps_ahead-1+t,'Close'],
                results.loc[fit_range+steps_ahead-1+t,'Pred']) )
    results = results[pd.notnull(results['Pred'])]
    error = np.sqrt(mean_squared_error(
        results.loc[(fit_range+steps_ahead-1):(len(series)-1),'Close'],
        results.loc[(fit_range+steps_ahead-1):(len(series)-1),'Pred']))
    return error, results

def gridSearch(series, model_selection, steps_ahead, fit_range, fastRMSE = False, numInstances=200) :
    '''
    Performs grid search of hyperparameters
    '''

    np.warnings.filterwarnings("ignore") # specify to ignore warning messages
    bestRMSE = math.inf
    gridResults = pd.DataFrame(columns=['Param', 'RMSE'])
    for param in model_selection[1]:
        try:
            model_sel = [model_selection[0], param]
            error, results = rollingEstimate(series, model_sel, steps_ahead = steps_ahead, fit_range = fit_range,
                                             verbose = False, fastRMSE = fastRMSE, dataPoints = dataPoints)
            if error < bestRMSE:
                bestRMSE = error
                bestParam = param
            gridResults = gridResults.append([{'Param':param, 'RMSE': error}], ignore_index=True)
            print('Method: {0} - Param: {1} - RMSE: {2:.4f}'.format(model_sel[0], param, error) )
        except:
            continue
    np.warnings.resetwarnings()
    print('Best parameters for method {}'.format(model_sel[0]))
    print('Param: {0} - RMSE: {1:.4f}'.format(bestParam, bestRMSE) )
    return bestParam, bestRMSE, gridResults
