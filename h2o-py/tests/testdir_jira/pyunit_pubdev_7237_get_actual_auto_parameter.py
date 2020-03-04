import sys
sys.path.insert(1,"../../")

from tests import pyunit_utils
import tests
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.xgboost import *
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.isolation_forest import H2OIsolationForestEstimator
from h2o.estimators.aggregator import H2OAggregatorEstimator
from h2o.estimators.kmeans import H2OKMeansEstimator
from h2o.estimators.naive_bayes import H2ONaiveBayesEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
import numpy as np


def test_set_auto_parameters_to_default_values():
    #testing default setup of following parameters:
    #distribution (available in Deep Learning, XGBoost, GBM):
    #stopping_metric (available in: GBM, DRF, Deep Learning, AutoML, XGBoost, Isolation Forest):
    #histogram_type (available in: GBM, DRF)
    #solver (available in: GLM) already done in hex.glm.GLM.defaultSolver()
    #categorical_encoding (available in: GBM, DRF, Deep Learning, K-Means, Aggregator, XGBoost, Isolation Forest)
    #fold_assignment (available in: GBM, DRF, Deep Learning, GLM, Naïve-Bayes, K-Means, XGBoost) for GLM done in pyunit_cv_cars_glm.py
    
    #Deep Learning:
    train_data = h2o.import_file(path=tests.locate("smalldata/gbm_test/ecology_model.csv"))
    train_data = train_data.drop('Site')
    train_data['Angaus'] = train_data['Angaus'].asfactor()

    test_data = h2o.import_file(path=tests.locate("smalldata/gbm_test/ecology_eval.csv"))
    test_data['Angaus'] = test_data['Angaus'].asfactor()

    dl1 = H2ODeepLearningEstimator(loss="CrossEntropy", epochs=1000, hidden=[20,20,20], seed=1234, reproducible=True, stopping_rounds=5)
    dl1.train(x=list(range(1, train_data.ncol)), y="Angaus", training_frame=train_data, validation_frame=test_data)

    dl2 = H2ODeepLearningEstimator(loss="CrossEntropy", epochs=1000, hidden=[20,20,20], seed=1234, reproducible=True, 
                                   distribution="bernoulli", categorical_encoding="OneHotInternal", stopping_rounds=5)
    dl2.train(x=list(range(1, train_data.ncol)), y="Angaus", training_frame=train_data, validation_frame=test_data)

    assert dl1.effective_params['distribution'] == dl2.actual_params['distribution']
    assert dl1.logloss() == dl2.logloss()
    assert dl1.effective_params['stopping_metric'] == dl2.effective_params['stopping_metric']
    assert dl1.effective_params['categorical_encoding'] == dl2.actual_params['categorical_encoding']
    assert dl1.effective_params['fold_assignment'] is None

    dl1 = H2ODeepLearningEstimator(loss="CrossEntropy", epochs=1000, hidden=[20,20,20], seed=1234, reproducible=True, nfolds=5)
    dl1.train(x=list(range(1,train_data.ncol)), y="Angaus", training_frame=train_data, validation_frame=test_data)

    dl2 = H2ODeepLearningEstimator(loss="CrossEntropy", epochs=1000, hidden=[20,20,20], seed=1234, reproducible=True, 
                                   distribution="bernoulli", categorical_encoding="OneHotInternal", nfolds=5, fold_assignment="Random")
    dl2.train(x=list(range(1,train_data.ncol)), y="Angaus", training_frame=train_data, validation_frame=test_data)

    assert dl1.effective_params['distribution'] == dl2.actual_params['distribution']
    assert dl1.logloss() == dl2.logloss()
    assert dl1.effective_params['stopping_metric'] is None
    assert dl1.effective_params['categorical_encoding'] == dl2.actual_params['categorical_encoding']
    assert dl1.effective_params['fold_assignment'] == dl2.actual_params['fold_assignment']
    

    #XGBoost:
    assert H2OXGBoostEstimator.available()

    prostate_frame = h2o.import_file(pyunit_utils.locate('smalldata/prostate/prostate.csv'))
    x = ['RACE']
    y = 'CAPSULE'
    prostate_frame[y] = prostate_frame[y].asfactor()
    prostate_frame.split_frame(ratios=[0.75], destination_frames=['prostate_training', 'prostate_validation'], seed=1)
    training_frame = h2o.get_frame('prostate_training')
    test_frame = h2o.get_frame('prostate_validation')

    xgb1 = H2OXGBoostEstimator(training_frame=training_frame, learn_rate=0.7, booster='gbtree', seed=1, ntrees=2, stopping_rounds=5)
    xgb1.train(x=x, y=y, training_frame=training_frame, validation_frame=test_frame)

    xgb2 = H2OXGBoostEstimator(training_frame=training_frame, learn_rate=0.7, booster='gbtree', seed=1, ntrees=2, distribution="bernoulli", 
                               categorical_encoding="OneHotInternal", stopping_rounds =5, stopping_metric='logloss')
    xgb2.train(x=x, y=y, training_frame=training_frame, validation_frame=test_frame)

    assert xgb1.effective_params['distribution'] == xgb2.actual_params['distribution']
    assert xgb1.logloss() == xgb2.logloss()
    assert xgb1.effective_params['stopping_metric'] == xgb2.actual_params['stopping_metric']
    assert xgb1.effective_params['categorical_encoding'] == xgb2.actual_params['categorical_encoding']
    assert xgb1.effective_params['fold_assignment'] is None


    xgb1 = H2OXGBoostEstimator(training_frame=training_frame, learn_rate=0.7, booster='gbtree', seed=1, ntrees=2, nfolds=5)
    xgb1.train(x=x, y=y, training_frame=training_frame)

    xgb2 = H2OXGBoostEstimator(training_frame=training_frame, learn_rate=0.7, booster='gbtree', seed=1, ntrees=2, distribution="bernoulli", 
                                         categorical_encoding="OneHotInternal", nfolds=5, fold_assignment="Random")
    xgb2.train(x=x, y=y, training_frame=training_frame)

    assert xgb1.effective_params['distribution'] == xgb2.actual_params['distribution']
    assert xgb1.logloss() == xgb2.logloss()
    assert xgb1.effective_params['stopping_metric'] is None
    assert xgb1.effective_params['categorical_encoding'] == xgb2.actual_params['categorical_encoding']
    assert xgb1.effective_params['fold_assignment'] == xgb2.actual_params['fold_assignment']


    #GBM:
    cars = h2o.import_file(path=pyunit_utils.locate("smalldata/junit/cars_20mpg.csv"))
    cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()
    cars["year"] = cars["year"].asfactor()
    predictors = ["displacement", "power", "weight", "acceleration", "year"]
    response = "economy_20mpg"
    train, valid = cars.split_frame(ratios=[.8], seed=1234)

    gbm1 = H2OGradientBoostingEstimator(seed=1234, stopping_rounds=3)
    gbm1.train(x=predictors, y=response, training_frame=train, validation_frame=valid)

    gbm2 = H2OGradientBoostingEstimator(seed=1234, stopping_rounds=3, distribution="bernoulli", stopping_metric="logloss", 
                                        histogram_type="UniformAdaptive", categorical_encoding="Enum")
    gbm2.train(x=predictors, y=response, training_frame=train, validation_frame=valid)

    assert gbm1.logloss() == gbm2.logloss()
    assert gbm1.effective_params['distribution'] == gbm2.actual_params['distribution']
    assert gbm1.effective_params['stopping_metric'] == gbm2.actual_params['stopping_metric']
    assert gbm1.effective_params['histogram_type'] == gbm2.actual_params['histogram_type']
    assert gbm1.effective_params['stopping_metric'] == gbm2.actual_params['stopping_metric']
    assert gbm1.effective_params['categorical_encoding'] == gbm2.actual_params['categorical_encoding']

    
    gbm1 = H2OGradientBoostingEstimator(seed = 1234, nfolds=5)
    gbm1.train(x=predictors, y=response, training_frame=train, validation_frame=valid)

    gbm2 = H2OGradientBoostingEstimator(seed = 1234, nfolds=5, fold_assignment='Random', distribution="bernoulli", 
                                        histogram_type="UniformAdaptive", categorical_encoding="Enum")
    gbm2.train(x=predictors, y=response, training_frame=train, validation_frame=valid)

    assert gbm1.logloss() == gbm2.logloss()
    assert gbm1.effective_params['distribution'] == gbm2.actual_params['distribution']
    assert gbm1.effective_params['stopping_metric'] is None
    assert gbm1.effective_params['histogram_type'] == gbm2.actual_params['histogram_type']
    assert gbm1.effective_params['fold_assignment'] == gbm2.actual_params['fold_assignment']
    assert gbm1.effective_params['categorical_encoding'] == gbm2.actual_params['categorical_encoding']
    
    
    #NAIVE BAYES:
    nb = H2ONaiveBayesEstimator(min_sdev=0.1, eps_sdev=0.5, seed=1234, nfolds=5)
    nb.train(x=predictors, y=response, training_frame=train)
    
    assert nb.effective_params['fold_assignment'] == "Random"
    

    #KMEANS:
    km1 = H2OKMeansEstimator(seed=1234, categorical_encoding="AUTO", nfolds=5)
    km1.train(x=["economy_20mpg", "displacement", "power", "weight", "acceleration", "year"], training_frame=cars)

    km2 = H2OKMeansEstimator(seed=1234, categorical_encoding="Enum", nfolds=5, fold_assignment='Random')
    km2.train(x=["economy_20mpg", "displacement", "power", "weight", "acceleration", "year"], training_frame=cars)

    assert km1.effective_params['categorical_encoding'] == km2.actual_params['categorical_encoding']
    assert km1.effective_params['fold_assignment'] == km2.actual_params['fold_assignment']
   
   
    #GBM:
    frame = h2o.import_file(path=pyunit_utils.locate("smalldata/logreg/prostate.csv"))
    frame.pop('ID')
    frame[frame['VOL'],'VOL'] = None
    frame[frame['GLEASON'],'GLEASON'] = None
    r = frame.runif()
    train = frame[r < 0.8]
    test = frame[r >= 0.8]
    
    gbm = H2OGradientBoostingEstimator(ntrees=5, max_depth=3)
    gbm.train(x=list(range(2,train.ncol)), y="CAPSULE", training_frame=train, validation_frame=test)
    
    assert gbm.effective_params['categorical_encoding'] is None
   
   
    #DRF:
    frame = h2o.import_file(path=pyunit_utils.locate("smalldata/gbm_test/ecology_model.csv"))
    frame["Angaus"] = frame["Angaus"].asfactor()
    frame["Weights"] = h2o.H2OFrame.from_python(abs(np.random.randn(frame.nrow, 1)).tolist())[0]
    train, calib = frame.split_frame(ratios=[.8], destination_frames=["eco_train", "eco_calib"], seed=42)

    rf1 = H2ORandomForestEstimator(ntrees=100, distribution="bernoulli", min_rows=10, max_depth=5, weights_column="Weights", 
                                   stopping_rounds = 3, calibrate_model=True, calibration_frame=calib, seed = 1234)
    rf1.train(x=list(range(2, train.ncol)), y="Angaus", training_frame=train)

    rf2 = H2ORandomForestEstimator(ntrees=100, distribution="bernoulli", min_rows=10, max_depth=5, weights_column="Weights",
                                   stopping_rounds = 3, stopping_metric='logloss', calibrate_model=True, calibration_frame=calib, 
                                   seed = 1234, categorical_encoding = 'Enum')
    rf2.train(x=list(range(2, train.ncol)), y="Angaus", training_frame=train)

    assert rf1.effective_params['stopping_metric'] == rf2.actual_params['stopping_metric']
    assert rf1.logloss() == rf2.logloss()
    assert rf1.effective_params['distribution'] == rf2.actual_params['distribution']
    assert rf1.effective_params['categorical_encoding'] == rf2.effective_params['categorical_encoding']

    rf1 = H2ORandomForestEstimator(ntrees=100, distribution="bernoulli", min_rows=10, max_depth=5, weights_column="Weights", 
                                   nfolds = 5, calibrate_model=True, calibration_frame=calib, seed = 1234)
    rf1.train(x=list(range(2, train.ncol)), y="Angaus", training_frame=train)

    rf2 = H2ORandomForestEstimator(ntrees=100, distribution="bernoulli", min_rows=10, max_depth=5, weights_column="Weights", 
                                   nfolds=5, fold_assignment='Random', calibrate_model=True, calibration_frame=calib, seed = 1234, 
                                   categorical_encoding = 'Enum')
    rf2.train(x=list(range(2, train.ncol)), y="Angaus", training_frame=train)

    assert rf1.effective_params['stopping_metric'] is None
    assert rf1.logloss() == rf2.logloss()
    assert rf1.effective_params['distribution'] == rf2.actual_params['distribution']
    assert rf1.effective_params['fold_assignment'] == rf2.actual_params['fold_assignment']
    assert rf1.effective_params['categorical_encoding'] == rf2.actual_params['categorical_encoding']


    #ISOLATION FORREST:
    train2 = h2o.import_file(pyunit_utils.locate("smalldata/anomaly/ecg_discord_train.csv"))

    if1 = H2OIsolationForestEstimator(ntrees=7, seed=12, sample_size=5, stopping_rounds=3)
    if1.train(training_frame=train2)

    if2 = H2OIsolationForestEstimator(ntrees=7, seed=12, sample_size=5, stopping_rounds=3, stopping_metric = 'anomaly_score', categorical_encoding="Enum")
    if2.train(training_frame=train2)

    assert if1.effective_params['stopping_metric'] == if2.actual_params['stopping_metric']
    assert if1._model_json['output']['training_metrics']._metric_json['mean_score'] == if2._model_json['output']['training_metrics']._metric_json['mean_score']
    assert if1.effective_params['categorical_encoding'] == if2.effective_params['categorical_encoding']


    #AGGREGATOR:
    frame = h2o.create_frame(rows=10000, cols=10, categorical_fraction=0.6, integer_fraction=0, binary_fraction=0, real_range=100,
        integer_range=100, missing_fraction=0, factors=100, seed=1234)
    
    agg1 = H2OAggregatorEstimator(target_num_exemplars=1000, rel_tol_num_exemplars=0.5, categorical_encoding="eigen")
    agg1.train(training_frame=frame)  
    
    agg2 = H2OAggregatorEstimator(target_num_exemplars=1000, rel_tol_num_exemplars=0.5)
    agg2.train(training_frame=frame)
    
    assert agg1.actual_params['categorical_encoding'] == agg2.actual_params['categorical_encoding']




if __name__ == "__main__":
    pyunit_utils.standalone_test(test_set_auto_parameters_to_default_values)
else:
    test_set_auto_parameters_to_default_values()
