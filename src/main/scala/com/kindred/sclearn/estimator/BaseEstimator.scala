package com.kindred.sclearn.estimator

import breeze.linalg.{DenseMatrix, DenseVector}

trait BaseEstimator{

  type T

  type Y <: BaseEstimator

  def fit(X: DenseMatrix[Double], y: DenseVector[T]):  Y

  def predict(X: DenseMatrix[Double]): DenseVector[T]

  def defaultScore: (DenseVector[T], DenseVector[T]) => Double

  def score(yPred: DenseVector[T], y: DenseVector[T],
            scoreFunc: (DenseVector[T], DenseVector[T]) => Double = defaultScore): Double

}
