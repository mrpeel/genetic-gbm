"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np

from keras.models import load_model
import lightgbm as lgb


import logging

def sc_mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            1.,
                                            None))
    return 100. * K.mean(diff, axis=-1)

def safe_log(input_array):
    return_vals = input_array.reshape(input_array.shape[0], ).copy()
    neg_mask = return_vals < 0
    return_vals = np.log(np.absolute(return_vals) + 1)
    return_vals[neg_mask] *= -1.
    return return_vals

def safe_exp(input_array):
    return_vals = input_array.reshape(input_array.shape[0], ).copy()
    neg_mask = return_vals < 0
    return_vals = np.exp(np.clip(np.absolute(return_vals), -7, 7)) - 1
    return_vals[neg_mask] *= -1.
    return return_vals


def safe_mape(actual_y, prediction_y):
    actual, prediction = reshape_vals(actual_y, prediction_y)
    diff = np.absolute((actual - prediction) / np.clip(np.absolute(actual), 1., None))
    return 100. * np.mean(diff)

def safe_maepe(actual_y, prediction_y):
    actual, prediction = reshape_vals(actual_y, prediction_y)
    mape = safe_mape(actual, prediction)
    mae = mean_absolute_error(actual, prediction)

    return (mape * mae)

def mle_eval(actual_y, eval_y):
    prediction_y = eval_y.get_label()
    assert len(actual_y) == len(prediction_y)
    return 'error', np.mean(np.absolute(safe_log(actual_y) - safe_log(prediction_y)))

def mape_eval(actual_y, eval_y):
    prediction_y = eval_y.get_label()
    assert len(actual_y) == len(prediction_y)
    return 'mape', safe_mape(actual_y, prediction_y), False

def maepe_eval(actual_y, eval_y):
    prediction_y = eval_y.get_label()
    assert len(actual_y) == len(prediction_y)
    return 'maepe', safe_maepe(actual_y, prediction_y), False

def reshape_vals(actual_y, prediction_y):
    actual_y = actual_y.reshape(actual_y.shape[0], )
    prediction_y = prediction_y.reshape(prediction_y.shape[0], )
    return actual_y, prediction_y

def mape_objective(actual_y, eval_y):
    predictions = eval_y.get_label()
    assert len(actual_y) == len(predictions)
    # diff = np.absolute((actual_y - predictions) / np.clip(np.absolute(actual_y), 1., None))
    # grad = -100. * diff
    # hess = np.ones(len(predictions))
    grad = np.sign(predictions - actual_y) / np.clip(np.absolute(predictions), 0.1, None)
    hess = np.zeros(len(predictions))
    return grad, hess

def round_down(num, divisor):
    return num - (num%divisor)

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
                              tree_method='hist',
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
    ae_all_train_x = pd.read_pickle('data/ae_train_x.pkl.gz', compression='gzip')
    df_all_train_y = pd.read_pickle('data/df_all_train_y.pkl.gz', compression='gzip')
    df_all_train_actuals = pd.read_pickle('data/df_all_train_actuals.pkl.gz', compression='gzip')
    df_all_test_x = pd.read_pickle('data/df_all_test_x.pkl.gz', compression='gzip')
    ae_all_test_x = pd.read_pickle('data/ae_test_x.pkl.gz', compression='gzip')
    df_all_test_y = pd.read_pickle('data/df_all_test_y.pkl.gz', compression='gzip')
    df_all_test_actuals = pd.read_pickle('data/df_all_test_actuals.pkl.gz', compression='gzip')

    train_y = df_all_train_y[0].values
    train_actuals = df_all_train_actuals[0].values
    train_log_y = safe_log(train_y)
    train_x = df_all_train_x.values
    train_ae = ae_all_train_x.values
    test_actuals = df_all_test_actuals.values
    test_y = df_all_test_y[0].values
    test_log_y = safe_log(test_y)
    test_x = df_all_test_x.values
    test_ae = ae_all_test_x.values

    # Use keras model to generate x vals
    # mae_intermediate_model = load_model('models/keras-mae-intermediate-model.h5')
    #
    # mae_vals_train = mae_intermediate_model.predict(train_x)
    # mae_vals_test = mae_intermediate_model.predict(test_x)

    # train_x = mae_vals_train
    # test_x = mae_vals_test

    model = compile_model(network)

    print('\rNetwork')

    for property in network:
        print(property, ':', network[property])
        logging.info('%s: %s' % (property, network[property]))


    x_train, x_test, y_train, y_test = train_test_split(train_ae, train_y, test_size=0.15)

    eval_set = [(x_test, y_test)]

    model.fit(x_train, y_train, early_stopping_rounds=5, eval_metric='mae', eval_set=eval_set,
                verbose=True)

    predictions = model.predict(test_ae)
    # inverse_predictions = safe_exp(predictions)
    score = mean_absolute_error(test_y, predictions)

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


    train_x = train_predictions.values
    train_y = train_actuals[0].values
    train_log_y = safe_log(train_y)
    test_x = test_predictions.values
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

def train_and_score_lgbm(network):

    df_all_train_x = pd.read_pickle('data/df_all_train_x.pkl.gz', compression='gzip')
    df_all_train_y = pd.read_pickle('data/df_all_train_y.pkl.gz', compression='gzip')
    df_all_train_actuals = pd.read_pickle('data/df_all_train_actuals.pkl.gz', compression='gzip')
    df_all_test_x = pd.read_pickle('data/df_all_test_x.pkl.gz', compression='gzip')
    df_all_test_y = pd.read_pickle('data/df_all_test_y.pkl.gz', compression='gzip')
    df_all_test_actuals = pd.read_pickle('data/df_all_test_actuals.pkl.gz', compression='gzip')

    train_y = df_all_train_y[0].values
    train_actuals = df_all_train_actuals[0].values
    train_log_y = safe_log(train_y)
    test_actuals = df_all_test_actuals[0].values
    test_y = df_all_test_y[0].values
    test_log_y = safe_log(test_y)

    # Use keras model to generate x vals
    #mae_intermediate_model = load_model('models/mae_intermediate_model.h5')
    # mae_vals_train = mae_intermediate_model.predict(train_x)
    # mae_vals_test = mae_intermediate_model.predict(test_x)
    #
    train_set = lgb.Dataset(df_all_train_x, label=train_y)
    eval_set = lgb.Dataset(df_all_test_x, reference=train_set, label=test_y)


    print('\rNetwork')

    for property in network:
        print(property, ':', network[property])
        logging.info('%s: %s' % (property, network[property]))

    params = network

    del params['number']

    params['verbosity'] = -1
    params['histogram_pool_size'] = 8192
    # params['metric'] = ['mae']
    params['metric_freq'] = 10

    # feature_name and categorical_feature
    gbm = lgb.train(params,
                    train_set,
                    valid_sets=eval_set,  # eval training data
                    feval=mape_eval,
                    fobj=mape_objective,
                    learning_rates=lambda iter: 0.05 * 0.999 ** (round_down(iter, 10)),
                    num_boost_round=500,
                    early_stopping_rounds=5)


    iteration_number = 500

    if gbm.best_iteration:
        iteration_number = gbm.best_iteration

    predictions = gbm.predict(df_all_test_x, num_iteration=iteration_number)
    eval_predictions = safe_exp(predictions)
    # eval_predictions = safe_exp(safe_exp(predictions))
    # eval_predictions = predictions

    mae = mean_absolute_error(test_actuals, eval_predictions)
    mape = safe_mape(test_actuals, eval_predictions)
    maepe = safe_maepe(test_actuals, eval_predictions)

    score = gbm.best_score['valid_0']['mape']

    print('\rResults')

    if np.isnan(score):
        score = 9999

    print('best iteration:', iteration_number)
    print('score:', score)
    print('mae:', mae)
    print('mape:', mape)
    print('maepe:', maepe)
    print('-' * 20)

    logging.info('best round: %d' % iteration_number)
    logging.info('score: %.4f' % score)
    logging.info('mae: %.4f' % mae)
    logging.info('mape: %.4f' % mape)
    logging.info('maepe: %.4f' % maepe)
    logging.info('-' * 20)

    return score

def train_and_score_xgb_ae():
    models['log_y'] = xgb.XGBRegressor(nthread=-1, n_estimators=500, max_depth=70, base_score=0.1,
                                       colsample_bylevel=0.7,
                                       colsample_bytree=1.0, gamma=0, learning_rate=0.025, min_child_weight=3)

    all_train_y = df_all_train_y.values
    all_train_log_y = safe_log(all_train_y)
    all_train_x = df_all_train_x.values
    all_test_actuals = df_all_test_actuals.values
    all_test_y = df_all_test_y.values
    all_test_x = df_all_test_x.values
    all_test_log_y = safe_log(all_test_y)

    mae_vals_train = keras_models['mae_intermediate_model'].predict(all_train_x)
    mae_vals_test = keras_models['mae_intermediate_model'].predict(all_test_x)
    eval_set = [(all_test_x, all_test_y)]
    models['log_y'].fit(all_train_x, all_train_y, early_stopping_rounds=25, eval_metric='mae', eval_set=eval_set,
                        verbose=True)