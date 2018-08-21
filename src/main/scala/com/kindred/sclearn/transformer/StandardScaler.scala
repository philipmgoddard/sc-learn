package com.kindred.sclearn.transformer

import breeze.linalg.{DenseMatrix, DenseVector}
import com.kindred.sclearn.estimator.BaseEstimator

// this not good. dont want to define type T at the level of the BaseEstimator.
// but this will cause problems with the SearchCV

class StandardScaler extends BaseTransformer  {
  override def transform(denseVector: DenseMatrix[Double]): DenseMatrix[Double] = ???

  override def fit(X: DenseMatrix[Double], y: Option[DenseVector[Double]]): BaseEstimator = ???

  override def predict(X: DenseMatrix[Double]): DenseVector[Double] = ???

  override protected[kindred] def run(paramMap: Map[String, Any]): BaseEstimator = ???

}
