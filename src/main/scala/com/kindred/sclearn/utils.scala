package com.kindred.sclearn

import breeze.linalg.{DenseMatrix, DenseVector}

object utils {

  def addBias(X: DenseMatrix[Double]): DenseMatrix[Double] = {
    val ones = DenseVector.fill(X.rows){1.0d}
    val Xbias = DenseMatrix.horzcat(new DenseMatrix(rows = X.rows, cols = 1, ones.toArray), X)
    Xbias
  }

}
