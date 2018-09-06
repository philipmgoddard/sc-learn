package com.kindred.sclearn.estimator

import breeze.linalg.{DenseMatrix, DenseVector}

trait BaseEstimator {

  def fit(X: DenseMatrix[Double], y: Option[DenseVector[Double]]): BaseEstimator

}
