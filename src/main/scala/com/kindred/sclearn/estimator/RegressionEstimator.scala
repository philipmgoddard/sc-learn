package com.kindred.sclearn.estimator

import breeze.linalg.{DenseMatrix, DenseVector}
import com.kindred.sclearn.metrics.RegressionMetrics.R2

trait RegressionEstimator extends BaseEstimator {

  type T = Double

  type Y = RegressionEstimator

  // fitting a estimator will return a new estimator with members updated so can score
  def fit(X: DenseMatrix[Double], y: DenseVector[Double]):  RegressionEstimator

  override def predict(X: DenseMatrix[Double]): DenseVector[Double]

  override def score(yPred: DenseVector[Double], y: DenseVector[Double],
                     scoreFunc: (DenseVector[Double], DenseVector[Double]) => Double = defaultScore): Double = {
    scoreFunc(yPred, y)
  }

  override def defaultScore: (DenseVector[Double], DenseVector[Double]) => Double = R2 _

}
