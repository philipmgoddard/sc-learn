package com.kindred.sclearn

import breeze.linalg.{DenseMatrix, DenseVector}

trait RegressionEstimator extends BaseEstimator {

  type T = Double
  type Y = RegressionPredictor

  def fit(X: DenseMatrix[Double], y: DenseVector[Double]): RegressionPredictor

}
