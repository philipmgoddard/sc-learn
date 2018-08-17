package com.kindred.sclearn.transformer

import breeze.linalg.{DenseMatrix, DenseVector}
import com.kindred.sclearn.estimator.BaseEstimator

// this not good. dont want to define type T at the level of the BaseEstimator.
// but this will cause problems with the SearchCV

class StandardScaler[T] extends BaseTransformer with BaseEstimator[T] {
  override def transform(denseVector: DenseMatrix[Double]): DenseMatrix[Double] = ???

  override protected[kindred] def run(paramMap: Map[String, Any]): BaseEstimator[T] = ???

}
