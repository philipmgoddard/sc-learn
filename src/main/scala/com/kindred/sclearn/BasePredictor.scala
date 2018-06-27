package com.kindred.sclearn

import breeze.linalg.{DenseMatrix, DenseVector}

trait BasePredictor {

  type T

  def predict(X: DenseMatrix[Double]): DenseVector[T]

  def defaultScore: (DenseVector[T], DenseVector[T]) => Double

  def score(yPred: DenseVector[T], y: DenseVector[T],
            scoreFunc: (DenseVector[T], DenseVector[T]) => Double = defaultScore): Double

}
