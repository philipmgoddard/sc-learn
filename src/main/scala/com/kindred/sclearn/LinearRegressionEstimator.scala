package com.kindred.sclearn

import breeze.linalg._
import RegressionMetrics.RMSE


class LinearRegressionEstimator(scoreFunc: (DenseVector[Double], DenseVector[Double]) => Double = RMSE)
  extends RegressionEstimator {

  private var w: Option[DenseVector[Double]] = None
  private var trainScore: Option[Double] = None

  override def fit(X: DenseMatrix[Double], y: DenseVector[Double]):  LinearRegressionEstimator = {

    // add a bias, and calculate coefficients using normal equation
    val ones = DenseVector.fill(X.rows){1.0}
    val Xbias = DenseMatrix.horzcat(new DenseMatrix(rows = X.rows, cols = 1, ones.toArray), X)
    val coef = {pinv(Xbias.t * Xbias)} * {Xbias.t} * y

    // create fitted estimator to be returned
    val trainedModel = new LinearRegressionEstimator()
    trainedModel.w = Some(coef)

    val trainPred = trainedModel.predict(X)
    trainedModel.trainScore = Some(trainedModel.score(trainPred, y, scoreFunc))

    trainedModel

  }

  // getter for coefficients
  def _coef: DenseVector[Double] = w match {
    case Some(c) => c
    case None => throw new Exception("Not fitted!")
  }

  // getter for score
  def _score: Double = trainScore match {
    case Some(s) => s
    case None => throw new Exception("Not fitted!")
  }


  override def predict(X: DenseMatrix[Double]): DenseVector[Double] = {

    val coef = w match {
      case Some(c) => c
      case None => throw new Exception("Not fitted!")
    }

    // add on the bias
    val ones = DenseVector.fill(X.rows){1.0}
    val Xbias = DenseMatrix.horzcat(new DenseMatrix(rows = X.rows, cols = 1, ones.toArray), X)

    // prediction_i =  sum_j (coef_j * X_ij)
    val Xw = Xbias(*, ::) :* coef
    sum(Xw(*, ::))
  }

}


object LinearRegressionEstimator {

  def apply(scoreFunc: (DenseVector[Double], DenseVector[Double]) => Double = RMSE): LinearRegressionEstimator =
    new LinearRegressionEstimator(scoreFunc)

}