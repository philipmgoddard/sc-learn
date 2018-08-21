package com.kindred.sclearn.estimator

import breeze.linalg.{DenseMatrix, DenseVector}
import com.kindred.sclearn.metrics.ClassificationMetrics.Accuracy

trait ClassificationEstimator extends BaseEstimator {

  def predictProb(X: DenseMatrix[Double]): DenseVector[Double]

  def predict(X: DenseMatrix[Double]): DenseVector[Double]

  def score(yPred: DenseVector[Double], y: DenseVector[Double],
                     scoreFunc: (DenseVector[Double], DenseVector[Double]) => Double = defaultScore): Double = {
    scoreFunc(yPred, y)
  }

   def defaultScore: (DenseVector[Double], DenseVector[Double]) => Double = Accuracy _
}
