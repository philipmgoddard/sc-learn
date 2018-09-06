package com.kindred.sclearn.estimator

import breeze.linalg.{DenseMatrix, DenseVector}
import com.kindred.sclearn.metrics.RegressionMetrics.R2

trait RegressionEstimator extends BaseModel {

   def score(yPred: DenseVector[Double], y: DenseVector[Double],
                     scoreFunc: (DenseVector[Double], DenseVector[Double]) => Double = defaultScore): Double = {
    scoreFunc(yPred, y)
  }

  def predict(X: DenseMatrix[Double]): DenseVector[Double]


   def defaultScore: (DenseVector[Double], DenseVector[Double]) => Double = R2 _

}
