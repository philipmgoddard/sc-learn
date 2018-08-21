package com.kindred.sclearn.linear_model

import breeze.linalg._
import breeze.numerics.{exp, log1p, sigmoid}
import breeze.optimize.FirstOrderMinimizer.OptParams
import breeze.optimize.{DiffFunction, OptimizationOption, minimize}
import com.kindred.sclearn.estimator.ClassificationEstimator
import com.kindred.sclearn.helpers.helpers.addBias


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
  override def fit(X: DenseMatrix[Double], y: Option[DenseVector[Double]]):  LogisticRegressionEstimator = {

    // add a bias if fitIntercept is true
    val Xb = if (fitIntercept) addBias(X) else X
    val yVal = y.get

    // define cost function
    def costFunction(coef: DenseVector[Double], X: DenseMatrix[Double]): Double = {
      val xBeta = X * coef
      val expXBeta = exp(xBeta)
      -1.0d * sum(yVal :* xBeta - log1p(expXBeta))
    }

    // define gradient of cost function
    def costFunctionGradient(coef: DenseVector[Double], X: DenseMatrix[Double]): DenseVector[Double] = {
      val xBeta = X * coef
      val probs = sigmoid(xBeta)
      X.t * (probs - yVal)
    }

    val optimalCoef = optimiseLinearModel(costFunction, costFunctionGradient, Xb, optOptions)
    val trainedModel = new LogisticRegressionEstimator(penalty, C, fitIntercept, tol, maxIter, alpha, randomState)
    trainedModel.w = Some(optimalCoef)
    trainedModel
  }

  override def predict(X: DenseMatrix[Double]): DenseVector[Double] = {
    val predProbs = predictProb(X)
    // check if > 0.5 - if yes 1, else 0
    breeze.numerics.I(predProbs :>= 0.5).toDenseVector
  }

  override def predictProb(X: DenseMatrix[Double]): DenseVector[Double] = {
    val Xb = if (fitIntercept) addBias(X) else X
    val XCoef = Xb(*, ::) :* coef_
    sigmoid(sum(XCoef(*, ::)))
  }

  // getter for coefficients
  override def coef_ : DenseVector[Double] = extractCoef(w, fitIntercept)

  // getter for intercept, if it is present
  override def intercept_ : Double = extractIntercept(w, fitIntercept)


  override def toString: String =
    s"LogisticRegressionEstimator(penalty=$penalty, C=$C, fitIntercept=$fitIntercept, tol=$tol, maxIter=$maxIter, alpha=$alpha, randomState=$randomState)"


    override protected[kindred] def run(paramMap: Map[String, Any]): LogisticRegressionEstimator = {
    LogisticRegressionEstimator.apply(paramMap)
  }

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

  def apply(paramMap: Map[String, Any]): LogisticRegressionEstimator = {
    new LogisticRegressionEstimator(
      penalty = paramMap.getOrElse("penalty", "l2").asInstanceOf[String],
      C = paramMap.getOrElse("C", 1.0).asInstanceOf[Double],
      fitIntercept = paramMap.getOrElse("fitIntercept", true).asInstanceOf[Boolean],
      tol = paramMap.getOrElse("tol", 1E-5).asInstanceOf[Double],
      maxIter = paramMap.getOrElse("maxIter", 1000).asInstanceOf[Int],
      alpha = paramMap.getOrElse("alpha", 0.5).asInstanceOf[Double],
      randomState = paramMap.getOrElse("randomState", 1234).asInstanceOf[Int])
  }


}

