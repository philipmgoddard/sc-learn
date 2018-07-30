package com.kindred.sclearn.linear_model

import breeze.linalg._
import com.kindred.sclearn.estimator.RegressionEstimator
import com.kindred.sclearn.helpers.helpers.addBias


class LinearRegressionEstimator(penalty: String, C: Double,
                                fitIntercept: Boolean, tol: Double,
                                maxIter: Integer, alpha: Double,
                                randomState: Integer)
  extends RegressionEstimator with LinearModel {

  // check penalty valid, initialise OptimizationOption object
  checkPenalty(penalty)
  private val optOptions = prepOptOptions(penalty, C, alpha, maxIter, tol, randomState)

  // fitted coefficients
  private var w: Option[DenseVector[Double]] = None

  /* some docstring for fit
   *
   * */
  override def fit(X: DenseMatrix[Double], y: DenseVector[Double]):  LinearRegressionEstimator = {

    val Xb = if (fitIntercept) addBias(X) else X

    // define cost function
    def costFunction(coef: DenseVector[Double], X: DenseMatrix[Double]): Double = {
      val yPred = sum(X * coef)
      1.0d / (2.0d * X.rows) *  scala.math.pow(sum(yPred - y), 2.0d)
    }

    // define gradient of cost function
    def costFunctionGradient(coef: DenseVector[Double], X: DenseMatrix[Double]): DenseVector[Double] = {
      val xCoef = X * coef
      X.t * (1.0d / X.rows) * (xCoef - y)
    }

    val optimalCoef = optimiseLinearModel(costFunction, costFunctionGradient, Xb, optOptions)
    val trainedModel = new LinearRegressionEstimator(penalty, C, fitIntercept, tol, maxIter, alpha, randomState)
    trainedModel.w = Some(optimalCoef)
    trainedModel
  }

  // predict using fitted coefficients
  override def predict(X: DenseMatrix[Double]): DenseVector[Double] = {
    if (fitIntercept) {
      addBias(X) * DenseVector.vertcat(DenseVector(_intercept), _coef)
    } else {
      X * _coef
    }
  }

  // getter for coefficients
  def _coef: DenseVector[Double] = extractCoef(w, fitIntercept)

  // getter for intercept, if it is present
  def _intercept: Double = extractIntercept(w, fitIntercept)

  // string representation
  override def toString: String =
    s"LinearRegressionEstimator(penalty=$penalty, C=$C, fitIntercept=$fitIntercept, tol=$tol, maxIter=$maxIter, alpha=$alpha, randomState=$randomState)"

}


object LinearRegressionEstimator {

  def apply(penalty: String = "l2",
            C: Double = 1.0,
            fitIntercept: Boolean = true,
            tol: Double = 1E-5,
            maxIter: Integer = 1000,
            alpha: Double = 0.5,
            randomState: Integer = 1234): LinearRegressionEstimator = {
    new LinearRegressionEstimator(
      penalty = penalty, C = C,
      fitIntercept = fitIntercept, tol = tol,
      maxIter = maxIter, alpha = alpha,
      randomState=randomState)
  }

//  def apply(penalty: Any = "l2",
//            C: Any = 1.0,
//            fitIntercept: Any = true,
//            tol: Any = 1E-5,
//            maxIter: Any = 1000,
//            alpha: Any = 0.5,
//            randomState: Any = 1234): LinearRegressionEstimator = {
//    new LinearRegressionEstimator(
//      penalty = penalty.asInstanceOf[String], C = C.asInstanceOf[Double],
//      fitIntercept = fitIntercept.asInstanceOf[Boolean], tol = tol.asInstanceOf[Double],
//      maxIter = maxIter.asInstanceOf[Int], alpha = alpha.asInstanceOf[Double],
//      randomState = randomState.asInstanceOf[Int])
//  }

  def apply(paramMap: Map[String, Any]): LinearRegressionEstimator = {
    new LinearRegressionEstimator(
      penalty = paramMap.getOrElse("penalty", "l2").asInstanceOf[String],
      C = paramMap.getOrElse("C", 1.0).asInstanceOf[Double],
      fitIntercept = paramMap.getOrElse("fitIntercept", true).asInstanceOf[Boolean],
      tol = paramMap.getOrElse("tol", 1E-5).asInstanceOf[Double],
      maxIter = paramMap.getOrElse("maxIter", 1000).asInstanceOf[Int],
      alpha = paramMap.getOrElse("alpha", 0.5).asInstanceOf[Double],
      randomState = paramMap.getOrElse("randomState", 1234).asInstanceOf[Int])
  }
}