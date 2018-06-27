package com.kindred.sclearn

import breeze.linalg.DenseVector
import RegressionMetrics._

object Test {

  val newObs = breeze.linalg.DenseMatrix((1.0, 4.3, 5.4), (1.2, 0.4, 0.3))
  val features = breeze.linalg.DenseMatrix((1.0, 4.0, 2.0), (6.3, 8.7, 2.3))
  val outcome = breeze.linalg.DenseVector(0.3, 5.0)
  val lr_predictor: LinearRegressionPredictor =  LinearRegressionEstimator().fit(features, outcome)

  val ypred: DenseVector[Double] = lr_predictor.predict(features)

//  val y: DenseVector[Double] = ???
//  val ypred: DenseVector[Double] = ???

  lr_predictor.score(ypred, outcome) // default: RMSE
  lr_predictor.score(ypred, outcome, SSE) // custom


}

/*
 * GOAL: linear classifier and regression. Random forest for regression and classification.
 * GridSearchCV, transformer, pipeline
 *
 *
 * */
// TODO: implememnt an optimisation scheme for linear classification and regression
// TODO: start implementing tests
// TODO implement logistic regression.
// TODO implement optimisation using breeze built in
// TODO implement gridsearchCV. Think about how can hold all training metrics (CV scores, variable importance etc) within
// TODO think how can do in parralel
// TODO implement feature transformers. Fit, transform, as per sklearn
// TODO implement pipelines. string together transformers, estimator to make something beautiful
// TODO implement a couple more models... RF? Neural net?
