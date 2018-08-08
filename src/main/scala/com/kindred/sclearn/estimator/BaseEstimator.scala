package com.kindred.sclearn.estimator

import breeze.linalg.{DenseMatrix, DenseVector}

trait BaseEstimator[T]{

  def fit(X: DenseMatrix[Double], y: DenseVector[T]): BaseEstimator[T]

  def predict(X: DenseMatrix[Double]): DenseVector[T]

  def defaultScore: (DenseVector[T], DenseVector[T]) => Double

  def score(yPred: DenseVector[T], y: DenseVector[T],
            scoreFunc: (DenseVector[T], DenseVector[T]) => Double = defaultScore): Double

  // protected method for use only in gridSearch. This is NOT part of public API
  // used to allow generic typing on the class
  // TODO: is run the best name for this? not really sure it describes what it does
  // literally just here because cannot have apply at this level
  protected[kindred] def run(paramMap: Map[String, Any]): BaseEstimator[T]

}
