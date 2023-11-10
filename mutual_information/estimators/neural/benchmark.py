import bmi


def neural_estimator_benchmark(X, Y, kwargs=None):
    estimators = []
    estimated_mi_arr = []

    if kwargs is None:
        # batch_size = 256, train_test_split = 0.2, learning_rate = 0.01, hidden_layers = (100, 100), verbose = True
        kwargs = {
            'batch_size': 256,
            'train_test_split': 0.2,
            'learning_rate': 0.01,
            'hidden_layers': (100, 100),
            'verbose': True
        }

    dve = bmi.estimators.DonskerVaradhanEstimator(**kwargs)
    dve_first = dve.estimate(X, Y)

    print('Donker Varadhan estimator: {:.2f}'.format(dve_first))
    estimated_mi_arr.append(dve_first)
    estimators.append(dve)

    mine = bmi.estimators.MINEEstimator(**kwargs)
    mine_first = mine.estimate(X, Y)

    print('MINE estimator: {:.2f}'.format(mine_first))
    estimated_mi_arr.append(mine_first)
    estimators.append(mine)

    infonce = bmi.estimators.InfoNCEEstimator(**kwargs)
    infonce_first = infonce.estimate(X, Y)

    print('InfoNCE estimator: {:.2f}'.format(infonce_first))
    estimated_mi_arr.append(infonce_first)
    estimators.append(infonce)

    nwj = bmi.estimators.NWJEstimator(**kwargs)
    nwj_first = nwj.estimate(X, Y)

    print('NWJ estimator: {:.2f}'.format(nwj_first))
    estimated_mi_arr.append(nwj_first)
    estimators.append(nwj)

    return estimators, estimated_mi_arr
