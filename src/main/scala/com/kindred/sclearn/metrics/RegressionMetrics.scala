package com.kindred.sclearn.metrics

import breeze.linalg._
import breeze.stats.mean

object RegressionMetrics {


  def SSE(yPred: DenseVector[Double], y: DenseVector[Double]): Double = {
    val sqrErr = (y - yPred) :* (y - yPred)
    sum(sqrErr)
  }

  def RMSE(yPred: DenseVector[Double], y: DenseVector[Double]): Double = {
    scala.math.sqrt(SSE(yPred, y))
  }

  // sum of total squares
  private def SSTO(y: DenseVector[Double]): Double = {
    val yAvg = mean(y)
    val STO = (y - yAvg) :* (y - yAvg)
    sum(STO)
  }

  def R2(yPred: DenseVector[Double], y: DenseVector[Double]) : Double = {
    1.0d - SSE(yPred, y) / SSTO(y)
  }

}
