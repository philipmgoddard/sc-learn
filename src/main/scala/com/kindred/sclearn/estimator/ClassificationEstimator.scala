package com.kindred.sclearn.estimator

import breeze.linalg.{DenseMatrix, DenseVector}
import com.kindred.sclearn.metrics.ClassificationMetrics.Accuracy

trait ClassificationEstimator extends BaseEstimator[Int] {

  def predictProb(X: DenseMatrix[Double]): DenseVector[Double]

  override def score(yPred: DenseVector[Int], y: DenseVector[Int],
                     scoreFunc: (DenseVector[Int], DenseVector[Int]) => Double = defaultScore): Double = {
    scoreFunc(yPred, y)
  }

  override def defaultScore: (DenseVector[Int], DenseVector[Int]) => Double = Accuracy _
}
