package com.kindred.sclearn.estimator

import breeze.linalg.{DenseMatrix, DenseVector}

trait BaseEstimator[T]{

//  type Y <: BaseEstimator[T]

  def fit(X: DenseMatrix[Double], y: DenseVector[T]):  BaseEstimator[T]

  def predict(X: DenseMatrix[Double]): DenseVector[T]

  def defaultScore: (DenseVector[T], DenseVector[T]) => Double

  def score(yPred: DenseVector[T], y: DenseVector[T],
            scoreFunc: (DenseVector[T], DenseVector[T]) => Double = defaultScore): Double

  protected[kindred] def run(paramMap: Map[String, Any]): BaseEstimator[T]

}
