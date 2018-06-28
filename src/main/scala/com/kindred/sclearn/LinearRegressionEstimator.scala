package com.kindred.sclearn

import breeze.linalg._

// is there a nice way this can be masked from the public facing API?
// make w private, default to None
class LinearRegressionEstimator(w: Option[DenseVector[Double]] = None) extends RegressionEstimator {

  // fit returns a new linear regression estimator object, which is actually fitted
  override def fit(X: DenseMatrix[Double], y: DenseVector[Double]): LinearRegressionEstimator = {

    // add a bias, and calculate coefficients using normal equation
    val ones = DenseVector.fill(X.rows){1.0}
    val Xbias = DenseMatrix.horzcat(new DenseMatrix(rows = X.rows, cols = 1, ones.toArray), X)
    val coef = {pinv(Xbias.t * Xbias)} * {Xbias.t} * y

    new LinearRegressionEstimator(w = Option(coef))

  }

  // getter for coefficients
  def _coef: DenseVector[Double] = w match {
    case Some(c) => c
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

  def apply() : LinearRegressionEstimator = new LinearRegressionEstimator()

}