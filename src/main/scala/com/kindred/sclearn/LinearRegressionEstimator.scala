package com.kindred.sclearn

import breeze.linalg._


class LinearRegressionEstimator extends RegressionEstimator {


  override def fit(X: DenseMatrix[Double], y: DenseVector[Double]): LinearRegressionPredictor = {


    // add a bias, and calculate coefficients using normal equation
    val ones = DenseVector.fill(X.rows){1.0}
    val Xbias = DenseMatrix.horzcat(new DenseMatrix(rows = X.rows, cols = 1, ones.toArray), X)
    val coef = {pinv(Xbias.t * Xbias)} * {Xbias.t} * y


    // return a com.kindred.sclearn.LinearRegressionPredictor
    val fittedPredictor = LinearRegressionPredictor(coef)

    // do i want to do anything more here? such as calculate the training error for the estimator?
    // this.trainingError type thing, so accessible as an attribute

    fittedPredictor

  }

}


object LinearRegressionEstimator {

  def apply() : LinearRegressionEstimator = new LinearRegressionEstimator()

}