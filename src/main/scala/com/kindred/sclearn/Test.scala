package com.kindred.sclearn

import breeze.linalg.{DenseVector, norm}
import RegressionMetrics._
import breeze.optimize._


object Test extends App {

  val newObs = breeze.linalg.DenseMatrix((1.0, 4.3, 5.4), (1.2, 0.4, 0.3))
  val features = breeze.linalg.DenseMatrix((1.0, 4.0, 2.0), (6.3, 8.7, 2.3))
  val outcome = breeze.linalg.DenseVector(0.3, 5.0)
  val lr_est =  LinearRegressionEstimator(scoreFunc = R2).fit(features, outcome)

  val ypred: DenseVector[Double] = lr_est.predict(features)

  //val lr_est2 = LinearRegressionEstimator()

//  val y: DenseVector[Double] = ???
//  val ypred: DenseVector[Double] = ???

  println(lr_est.score(ypred, outcome)) // default: RMSE
  println(lr_est.score(ypred, outcome, R2)) // custom

  println(lr_est._coef)
  println(lr_est._score)
  println(lr_est)



  val f = new DiffFunction[DenseVector[Double]] {
    def calculate(x: DenseVector[Double]) = {
      (breeze.linalg.norm((x - 3d) :^ 2d,1d),(x * 2d) - 6d);
    }
  }

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
