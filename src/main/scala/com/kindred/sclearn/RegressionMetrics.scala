package com.kindred.sclearn

import breeze.linalg._

object RegressionMetrics {

  def SSE(yPred: DenseVector[Double], y: DenseVector[Double]): Double = {
    val sqrErr = (y - yPred) :* (y - yPred)
    sum(sqrErr)
  }

  def RMSE(yPred: DenseVector[Double], y: DenseVector[Double]): Double = {
    scala.math.sqrt(SSE(yPred, y))
  }

}
