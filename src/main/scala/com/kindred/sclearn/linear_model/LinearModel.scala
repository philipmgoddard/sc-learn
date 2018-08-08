package com.kindred.sclearn.linear_model

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.optimize.FirstOrderMinimizer.OptParams
import breeze.optimize.{DiffFunction, OptimizationOption, minimize}


trait LinearModel  {

  def checkPenalty(penalty: String) : Unit = {
    if ((penalty != "l1") & (penalty != "l2")) throw new Exception("Only penalty = l1 or l2 allowed")
  }

  def prepOptOptions(penalty: String, C: Double, alpha: Double, maxIter: Int, tol: Double, randomState: Int) : OptimizationOption = {
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

  def optimiseLinearModel(costFunction: (DenseVector[Double], DenseMatrix[Double]) => Double,
                          costFunctionGradient: (DenseVector[Double], DenseMatrix[Double]) => DenseVector[Double],
                          X: DenseMatrix[Double],
                          optOptions: OptimizationOption): DenseVector[Double]  = {

    // define breeze DiffFunction
    val f = new DiffFunction[DenseVector[Double]] {
      def calculate(coef: DenseVector[Double]): (Double, DenseVector[Double]) = {
        (costFunction(coef, X), costFunctionGradient(coef, X))
      }
    }

    // perform the optimisation
    val optimalCoef = minimize(fn = f,
      init = DenseVector.fill(X.cols){0.0d},
      options = optOptions)

    optimalCoef
  }

  // extract coefficients
  // note that if intercept fitted, 0th is the bias (intercept)
  def extractCoef(w: Option[DenseVector[Double]], fitIntercept: Boolean): DenseVector[Double] = w match {
    case Some(c) =>
      val wSize = c.size
      if (fitIntercept) c(1 until wSize) else c
    case None => throw new Exception("Not fitted!")
  }

  def extractIntercept(w: Option[DenseVector[Double]], fitIntercept: Boolean): Double = w match {
    case Some(c) =>
      if (fitIntercept) c(0) else throw new Exception("fitIntercept = false")
    case None => throw new Exception("Not fitted!")
  }

  def coef_ : DenseVector[Double]

  def intercept_ : Double

}
