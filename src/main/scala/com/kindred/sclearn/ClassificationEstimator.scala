package com.kindred.sclearn

import breeze.linalg.{DenseMatrix, DenseVector}

trait ClassificationEstimator extends BaseEstimator {

  type T = Int

  type Y = BasePredictor

  def fit(X: DenseMatrix[Double], y: DenseVector[Int]): BasePredictor
}
