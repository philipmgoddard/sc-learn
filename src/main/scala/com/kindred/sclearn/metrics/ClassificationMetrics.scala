package com.kindred.sclearn.metrics

import breeze.linalg._


object ClassificationMetrics {

  def Accuracy(yPred: DenseVector[Double], y: DenseVector[Double]): Double = {
    // TODO: check what is returned by :== operator
    (yPred :== y).length / yPred.length
  }

}
