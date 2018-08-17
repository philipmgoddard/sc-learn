package com.kindred.sclearn.estimator

import breeze.linalg.{DenseMatrix, DenseVector}
import com.kindred.sclearn.metrics.ClassificationMetrics.Accuracy

trait ClassificationEstimator extends BaseEstimator[Int] {

  def predictProb(X: DenseMatrix[Double]): DenseVector[Double]

  def predict(X: DenseMatrix[Double]): DenseVector[Int]

  def fit(X: DenseMatrix[Double], y: DenseVector[Int]): BaseEstimator[Int]


  def score(yPred: DenseVector[Int], y: DenseVector[Int],
                     scoreFunc: (DenseVector[Int], DenseVector[Int]) => Double = defaultScore): Double = {
    scoreFunc(yPred, y)
  }

   def defaultScore: (DenseVector[Int], DenseVector[Int]) => Double = Accuracy _
}
