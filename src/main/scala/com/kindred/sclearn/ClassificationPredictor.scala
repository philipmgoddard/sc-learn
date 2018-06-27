package com.kindred.sclearn

import breeze.linalg.{DenseMatrix, DenseVector}

trait ClassificationPredictor extends BasePredictor {

  type T = Int

  def predict(X: DenseMatrix[Double]): DenseVector[Int]

}
