package com.kindred.sclearn.helpers

import breeze.linalg.{DenseMatrix, DenseVector}

object helpers {

  def addBias(X: DenseMatrix[Double]): DenseMatrix[Double] = {
    val ones = DenseVector.fill(X.rows){1.0d}
    val Xbias = DenseMatrix.horzcat(new DenseMatrix(rows = X.rows, cols = 1, ones.toArray), X)
    Xbias
  }

}
