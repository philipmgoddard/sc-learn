package com.kindred.sclearn.estimator

import breeze.linalg.{DenseMatrix, DenseVector}

trait BaseModel extends BaseEstimator {

  override def fit(X: DenseMatrix[Double], y: Option[DenseVector[Double]]): BaseModel

  def predict(X: DenseMatrix[Double]): DenseVector[Double]


  // protected method for use only in gridSearch. This is NOT part of public API
  // used to allow generic typing on the class
  // not the most elegant... wonder if there is a way to make nicer...
  protected[kindred] def run(paramMap: Map[String, Any]): BaseModel

}
