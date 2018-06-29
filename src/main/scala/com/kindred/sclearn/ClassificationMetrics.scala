package com.kindred.sclearn

import breeze.linalg._


object ClassificationMetrics {

  def Accuracy(yPred: DenseVector[Int], y: DenseVector[Int]): Double = {
    // TODO: check what is returned by :== operator
    (yPred :== y).length / yPred.length
  }

}
