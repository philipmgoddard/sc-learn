package com.kindred.sclearn

import breeze.linalg.{DenseMatrix, DenseVector}
import RegressionMetrics.RMSE

trait RegressionPredictor extends BasePredictor {

  type T = Double

  override def predict(X: DenseMatrix[Double]): DenseVector[Double]

  override def score(yPred: DenseVector[Double], y: DenseVector[Double],
                     scoreFunc: (DenseVector[Double], DenseVector[Double]) => Double = defaultScore): Double = {
    scoreFunc(yPred, y)
  }

  override def defaultScore: (DenseVector[Double], DenseVector[Double]) => Double = RMSE _

}
