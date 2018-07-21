package com.kindred.sclearn.linear_model

import breeze.linalg._
import breeze.numerics.{exp, log1p, sigmoid}
import breeze.optimize.FirstOrderMinimizer.OptParams
import breeze.optimize.{DiffFunction, OptimizationOption, minimize}
import com.kindred.sclearn.estimator.ClassificationEstimator
import com.kindred.sclearn.helpers.helpers.addBias


// todo: a lot of duplicate code between Linear and Logistic estimators- BaseLinearModel and then inherit?
class LogisticRegressionEstimator(penalty: String, C: Double,
                                  fitIntercept: Boolean, tol: Double,
                                  maxIter: Integer, alpha: Double,
                                  randomState: Integer)
  extends ClassificationEstimator with LinearModel {

  // check penalty valid, initialise OptimizationOption object
  checkPenalty(penalty)
  private val optOptions = prepOptOptions(penalty, C, alpha, maxIter, tol, randomState)

  // fitted coefficients
  private var w: Option[DenseVector[Double]] = None

  /* some docstring
   *
   * */
  override def fit(X: DenseMatrix[Double], y: DenseVector[Int]):  LogisticRegressionEstimator = {

    // add a bias if fitIntercept is true
    val Xb = if (fitIntercept) addBias(X) else X

    // define cost function
    def costFunction(coef: DenseVector[Double], X: DenseMatrix[Double]): Double = {
      val xBeta = X * coef
      val expXBeta = exp(xBeta)
      -1.0d * sum((convert(y, Double) :* xBeta) - log1p(expXBeta))
    }

    // define gradient of cost function
    def costFunctionGradient(coef: DenseVector[Double], X: DenseMatrix[Double]): DenseVector[Double] = {
      val xBeta = X * coef
      val probs = sigmoid(xBeta)
      X.t * (probs - convert(y, Double))
    }

    val optimalCoef = optimiseLinearModel(costFunction, costFunctionGradient, Xb, optOptions)
    val trainedModel = new LogisticRegressionEstimator(penalty, C, fitIntercept, tol, maxIter, alpha, randomState)
    trainedModel.w = Some(optimalCoef)
    trainedModel
  }

  override def predict(X: DenseMatrix[Double]): DenseVector[Int] = {
    val predProbs = predictProb(X)
    // check if > 0.5 - if yes 1, else 0
    convert(breeze.numerics.I(predProbs :>= 0.5).toDenseVector, Int)
  }

  override def predictProb(X: DenseMatrix[Double]): DenseVector[Double] = {
    val Xb = if (fitIntercept) addBias(X) else X
    val XCoef = Xb(*, ::) :* _coef
    sigmoid(sum(XCoef(*, ::)))
  }

  // getter for coefficients
  def _coef: DenseVector[Double] = extractCoef(w, fitIntercept)

  // getter for intercept, if it is present
  def _intercept: Double = extractIntercept(w, fitIntercept)

}


object LogisticRegressionEstimator {

  def apply(penalty: String = "l2",
            C: Double = 1.0,
            fitIntercept: Boolean = true,
            tol: Double = 1E-5,
            maxIter: Integer = 1000,
            alpha: Double = 0.5,
            randomState: Integer = 1234): LogisticRegressionEstimator = {

    new LogisticRegressionEstimator(
      penalty = penalty, C = C,
      fitIntercept = fitIntercept, tol = tol,
      maxIter = maxIter, alpha = alpha,
      randomState=randomState)
  }


}

