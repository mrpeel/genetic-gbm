"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
import xgboost as xgb

from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np

from keras.models import Sequential, Model
from keras.models import load_model


import logging

def sc_mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            1.,
                                            None))
    return 100. * K.mean(diff, axis=-1)

def safe_log(input_array):
    return_vals = input_array.copy()
    neg_mask = return_vals < 0
    return_vals = np.log(np.absolute(return_vals) + 1)
    return_vals[neg_mask] *= -1.
    return return_vals

def safe_exp(input_array):
    return_vals = input_array.copy()
    neg_mask = return_vals < 0
    return_vals = np.exp(np.clip(np.absolute(return_vals), -7, 7)) - 1
    return_vals[neg_mask] *= -1.
    return return_vals


def safe_mape(actual_y, prediction_y):
    """
    Calculate mean absolute percentage error

    Args:
        actual_y - numpy array containing targets with shape (n_samples, n_targets)
        prediction_y - numpy array containing predictions with shape (n_samples, n_targets)
    """
    diff = np.absolute((actual_y - prediction_y) / np.clip(np.absolute(actual_y), 1., None))
    return 100. * np.mean(diff)

def safe_maepe(actual_y, prediction_y):
    mape = safe_mape(actual_y, prediction_y)
    mae = mean_absolute_error(actual_y, prediction_y)

    return (mape * mae)

def compile_model(network):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    max_depth = network['max_depth']
    base_score = network['base_score']
    colsample_bylevel = network['colsample_bylevel']
    colsample_bytree = network['colsample_bytree']
    gamma = network['gamma']
    learning_rate = network['learning_rate']
    booster = network['learning_rate']
    min_child_weight = network['min_child_weight']

    model  = xgb.XGBRegressor(nthread=-1, n_estimators=5000,
                              # booster=booster,
                              max_depth=max_depth,
                              base_score=base_score,
                              colsample_bylevel=colsample_bylevel,
                              colsample_bytree=colsample_bytree,
                              gamma=gamma,
                              learning_rate=learning_rate,
                               min_child_weight=min_child_weight)

    return model

def train_and_score_xgb(network):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network

    """

    df_all_train_x = pd.read_pickle('data/df_all_train_x.pkl.gz', compression='gzip')
    df_all_train_y = pd.read_pickle('data/df_all_train_y.pkl.gz', compression='gzip')
    df_all_train_actuals = pd.read_pickle('data/df_all_train_actuals.pkl.gz', compression='gzip')
    df_all_test_x = pd.read_pickle('data/df_all_test_x.pkl.gz', compression='gzip')
    df_all_test_y = pd.read_pickle('data/df_all_test_y.pkl.gz', compression='gzip')
    df_all_test_actuals = pd.read_pickle('data/df_all_test_actuals.pkl.gz', compression='gzip')

    train_y = df_all_train_y[0].values
    train_actuals = df_all_train_actuals[0].values
    train_log_y = safe_log(train_y)
    train_x = df_all_train_x.as_matrix()
    test_actuals = df_all_test_actuals.as_matrix()
    test_y = df_all_test_y[0].values
    test_log_y = safe_log(test_y)
    test_x = df_all_test_x.as_matrix()

    # Use keras model to generate x vals
    mae_intermediate_model = load_model('models/mae_intermediate_model.h5')

    mae_vals_train = mae_intermediate_model.predict(train_x)
    mae_vals_test = mae_intermediate_model.predict(test_x)

    train_x = mae_vals_train
    test_x = mae_vals_test

    model = compile_model(network)

    print('\rNetwork')

    for property in network:
        print(property, ':', network[property])
        logging.info('%s: %s' % (property, network[property]))


    eval_set = [(test_x, test_log_y)]
    model.fit(train_x, train_log_y, early_stopping_rounds=5, eval_metric='mae', eval_set=eval_set,
                verbose=False)

    predictions = model.predict(test_x)
    score = mean_absolute_error(test_log_y, predictions)

    print('\rResults')

    best_round = model.best_iteration

    if np.isnan(score):
        score = 9999

    print('best round:', best_round)
    print('loss:', score)
    print('-' * 20)

    logging.info('best round: %d' % best_round)
    logging.info('loss: %.4f' % score)
    logging.info('-' * 20)

    return score

def train_and_score_bagging(network):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network

    """

    train_predictions = pd.read_pickle('data/train_predictions.pkl.gz', compression='gzip')
    test_predictions = pd.read_pickle('data/test_predictions.pkl.gz', compression='gzip')

    train_actuals = pd.read_pickle('data/train_actuals.pkl.gz', compression='gzip')
    test_actuals = pd.read_pickle('data/test_actuals.pkl.gz', compression='gzip')


    train_x = train_predictions.as_matrix()
    train_y = train_actuals[0].values
    train_log_y = safe_log(train_y)
    test_x = test_predictions.as_matrix()
    test_y = test_actuals[0].values
    test_log_y = safe_log(test_y)

    model = compile_model(network)

    print('\rNetwork')

    for property in network:
        print(property, ':', network[property])
        logging.info('%s: %s' % (property, network[property]))


    eval_set = [(test_x, test_log_y)]
    model.fit(train_x, train_log_y, early_stopping_rounds=20, eval_metric='mae', eval_set=eval_set,
                verbose=False)

    predictions = model.predict(test_x)
    inverse_predictions = safe_exp(predictions)
    mae = mean_absolute_error(test_y, inverse_predictions)
    mape = safe_mape(test_y, inverse_predictions)
    maepe = safe_maepe(test_y, inverse_predictions)

    score = maepe

    print('\rResults')

    best_round = model.best_iteration

    if np.isnan(score):
        score = 9999

    print('best round:', best_round)
    print('mae:', mae)
    print('mape:', mape)
    print('maepe:', maepe)
    print('-' * 20)

    logging.info('best round: %d' % best_round)
    logging.info('mae: %.4f' % mae)
    logging.info('mape: %.4f' % mape)
    logging.info('maepe: %.4f' % maepe)
    logging.info('-' * 20)

    return score
