package com.kindred.sclearn

import breeze.linalg.{DenseMatrix, DenseVector}

trait BaseEstimator{

  type T

  type Y <: BasePredictor

  def fit(X: DenseMatrix[Double], y: DenseVector[T]): Y

}
