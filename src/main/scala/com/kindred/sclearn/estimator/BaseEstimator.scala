package com.kindred.sclearn.estimator

import breeze.linalg.{DenseMatrix, DenseVector}


// one possibility- base estimator is no longer types. everything must be on Double

trait BaseEstimator {

  def fit(X: DenseMatrix[Double], y: Option[DenseVector[Double]]): BaseEstimator

  // TODO: I do not want this here. Is it possible to push down?
  // Essentially want SearchCV to take Classificatoin or Regression Estimators
  def predict(X: DenseMatrix[Double]): DenseVector[Double]


  // protected method for use only in gridSearch. This is NOT part of public API
  // used to allow generic typing on the class
  // TODO: is there a way to get rid of this??
  protected[kindred] def run(paramMap: Map[String, Any]): BaseEstimator

}
