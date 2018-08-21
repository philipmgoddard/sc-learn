package com.kindred.sclearn.transformer

import breeze.linalg.DenseMatrix
import com.kindred.sclearn.estimator.BaseEstimator

trait BaseTransformer extends BaseEstimator {

  def transform(DenseVector: DenseMatrix[Double]): DenseMatrix[Double]

}
