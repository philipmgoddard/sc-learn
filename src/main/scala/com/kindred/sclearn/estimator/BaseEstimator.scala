package com.kindred.sclearn.estimator

import breeze.linalg.{DenseMatrix, DenseVector}


// one possibility- base estimator is no longer types. everything must be on Double

trait BaseEstimator {

  def fit(X: DenseMatrix[Double], y: Option[DenseVector[Double]]): BaseEstimator


}
