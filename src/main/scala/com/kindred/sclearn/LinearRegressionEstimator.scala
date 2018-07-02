package com.kindred.sclearn

import breeze.linalg._
import RegressionMetrics.RMSE
import breeze.optimize.{DiffFunction, L2Regularization, OptimizationOption, minimize}
import com.kindred.sclearn.utils.addBias


class LinearRegressionEstimator(scoreFunc: (DenseVector[Double], DenseVector[Double]) => Double,
                                optOptions: OptimizationOption*)
  extends RegressionEstimator {

  private var w: Option[DenseVector[Double]] = None
  private var trainScore: Option[Double] = None

  override def fit(X: DenseMatrix[Double], y: DenseVector[Double]):  LinearRegressionEstimator = {

    // add bias
    val Xbias = addBias(X)

    // define cost function
    def costFunction(coef: DenseVector[Double]): Double = {
      val yPred = sum(Xbias * coef)
      1.0d / (2.0d * Xbias.rows) *  scala.math.pow(sum(yPred - y), 2.0d)
    }

    // define gradient of cost function
    def costFunctionGradient(coef: DenseVector[Double]): DenseVector[Double] = {
      val xCoef = Xbias * coef
      Xbias.t * (1.0d / Xbias.rows) * (xCoef - y)
    }

    // define breeze DiffFunction
    val f = new DiffFunction[DenseVector[Double]] {
      def calculate(coef: DenseVector[Double]): (Double, DenseVector[Double]) = {
        (costFunction(coef), costFunctionGradient(coef))
      }
    }

    // optimisation - uses LBFGS by default. pass in variable args to pass to minimizer
    val optimalCoef = minimize(f, DenseVector.fill(Xbias.cols){0.0d}, optOptions: _*)

    // create fitted estimator to be returned
    val trainedModel = new LinearRegressionEstimator(scoreFunc)
    trainedModel.w = Some(optimalCoef)

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


  // predict using fitted coefficients
  override def predict(X: DenseMatrix[Double]): DenseVector[Double] = {
    val Xbias = addBias(X)
    Xbias * _coef
  }

}


object LinearRegressionEstimator {

  def apply(scoreFunc: (DenseVector[Double], DenseVector[Double]) => Double = RMSE,
            optOptions: List[OptimizationOption] = List(L2Regularization(0.0001d)) ): LinearRegressionEstimator = {
    new LinearRegressionEstimator(scoreFunc, optOptions: _*)
  }

}