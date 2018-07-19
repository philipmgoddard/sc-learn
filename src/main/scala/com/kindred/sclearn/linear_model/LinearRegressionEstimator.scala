package com.kindred.sclearn.linear_model

import breeze.linalg._
import breeze.optimize.FirstOrderMinimizer.OptParams
import breeze.optimize.{DiffFunction, OptimizationOption, minimize}
import com.kindred.sclearn.estimator.RegressionEstimator
import com.kindred.sclearn.helpers.helpers.addBias

// todo: a lot of duplicate code between Linear and Logistic estimators- BaseLinearModel and then inherit?
class LinearRegressionEstimator(penalty: String, C: Double,
                                fitIntercept: Boolean, tol: Double,
                                maxIter: Integer, alpha: Double,
                                randomState: Integer)
  extends RegressionEstimator {

  if ((penalty != "l1") & (penalty != "l2")) throw new Exception("Only penalty = l1 or l2 allowed")

  // fitted coefficients
  private var w: Option[DenseVector[Double]] = None

  // set up the optimisation options to define L1 or L2 penalties
  private val optOptions: OptimizationOption = {
    OptimizationOption.fromOptParams(
      OptParams(
        batchSize = 512,
        regularization = 1.0 / C,
        alpha = alpha,
        maxIterations = maxIter,
        useL1 = if (penalty == "l1") true else false,
        tolerance = tol,
        useStochastic = false,
        randomSeed = randomState
      )
    )
  }

  override def fit(X: DenseMatrix[Double], y: DenseVector[Double]):  LinearRegressionEstimator = {

    // add bias
    val Xb = if (fitIntercept) addBias(X) else X

    // define cost function
    def costFunction(coef: DenseVector[Double]): Double = {
      val yPred = sum(Xb * coef)
      1.0d / (2.0d * Xb.rows) *  scala.math.pow(sum(yPred - y), 2.0d)
    }

    // define gradient of cost function
    def costFunctionGradient(coef: DenseVector[Double]): DenseVector[Double] = {
      val xCoef = Xb * coef
      Xb.t * (1.0d / Xb.rows) * (xCoef - y)
    }

    // define breeze DiffFunction
    val f = new DiffFunction[DenseVector[Double]] {
      def calculate(coef: DenseVector[Double]): (Double, DenseVector[Double]) = {
        (costFunction(coef), costFunctionGradient(coef))
      }
    }

    // optimisation
    val optimalCoef = minimize(fn = f,
      init = DenseVector.fill(Xb.cols){0.0d},
      options = optOptions)

    // create fitted estimator to be returned
    val trainedModel = new LinearRegressionEstimator(penalty, C, fitIntercept, tol, maxIter, alpha, randomState)
    trainedModel.w = Some(optimalCoef)

    trainedModel
  }

  // getter for coefficients
  def _coef: DenseVector[Double] = w match {
    case Some(c) =>
      val wSize = w.size
      if (fitIntercept) c(1 until wSize) else c
    case None => throw new Exception("Not fitted!")
  }

  // getter for intercept, if it is present
  def _intercept: Double = w match {
    case Some(c) =>
      if (fitIntercept) c(0) else throw new Exception("fitIntercept = false")
    case None => throw new Exception("Not fitted!")
  }

  // predict using fitted coefficients
  override def predict(X: DenseMatrix[Double]): DenseVector[Double] = {
    val Xb = if (fitIntercept) addBias(X) else X
    Xb * _coef
  }

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

}