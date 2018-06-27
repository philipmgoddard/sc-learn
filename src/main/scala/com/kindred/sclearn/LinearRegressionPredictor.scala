package com.kindred.sclearn

import breeze.linalg._

class LinearRegressionPredictor(w: DenseVector[Double]) extends RegressionPredictor {


  // getter for coefficients
  def coef: DenseVector[Double] = w


  override def predict(X: DenseMatrix[Double]): DenseVector[Double] = {

    // add on the bias
    val ones = DenseVector.fill(X.rows){1.0}
    val Xbias = DenseMatrix.horzcat(new DenseMatrix(rows = X.rows, cols = 1, ones.toArray), X)

    // prediction_i =  sum_j (coef_j * X_ij)
    val Xw = Xbias(*, ::) :* w
    sum(Xw(*, ::))
  }


}


object LinearRegressionPredictor {

  def apply(w: DenseVector[Double]): LinearRegressionPredictor = new LinearRegressionPredictor(w)

}



