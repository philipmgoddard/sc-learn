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
  extends ClassificationEstimator {

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

  override def fit(X: DenseMatrix[Double], y: DenseVector[Int]):  LogisticRegressionEstimator = {

    // add a bias if fitIntercept is true
    val Xb = if (fitIntercept) addBias(X) else X

    // define cost function
    def costFunction(coef: DenseVector[Double]): Double = {
      val xBeta = Xb * coef
      val expXBeta = exp(xBeta)
      -1.0d * sum((convert(y, Double) :* xBeta) - log1p(expXBeta))
    }

    // define gradient of cost function
    def costFunctionGradient(coef: DenseVector[Double]): DenseVector[Double] = {
      val xBeta = Xb * coef
      val probs = sigmoid(xBeta)
      Xb.t * (probs - convert(y, Double))
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

