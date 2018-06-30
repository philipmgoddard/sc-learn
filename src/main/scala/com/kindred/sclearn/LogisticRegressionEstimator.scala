package com.kindred.sclearn

import ClassificationMetrics.Accuracy
import breeze.linalg._
import breeze.numerics.exp
import breeze.numerics.log1p
import breeze.numerics.sigmoid
import breeze.optimize.DiffFunction
import breeze.optimize.minimize



class LogisticRegressionEstimator(scoreFunc: (DenseVector[Int], DenseVector[Int]) => Double = Accuracy)
  extends ClassificationEstimator {

  private var w: Option[DenseVector[Double]] = None
  private var trainScore: Option[Double] = None

  override def fit(X: DenseMatrix[Double], y: DenseVector[Int]):  LogisticRegressionEstimator = {

    // add a bias
    val ones = DenseVector.fill(X.rows){1.0d}
    val Xbias = DenseMatrix.horzcat(new DenseMatrix(rows = X.rows, cols = 1, ones.toArray), X)

    // define cost function
    def costFunction(coef: DenseVector[Double]): Double = {
      val xBeta = Xbias * coef
      val expXBeta = exp(xBeta)
      -1.0d * sum((convert(y, Double) :* xBeta) - log1p(expXBeta))
    }

    // define gradient of cost function
    def costFunctionGradient(coef: DenseVector[Double]): DenseVector[Double] = {
      val xBeta = Xbias * coef
      val probs = sigmoid(xBeta)
      Xbias.t * (probs - convert(y, Double))
    }

    // define breeze DiffFunction
    val f = new DiffFunction[DenseVector[Double]] {
       def calculate(coef: DenseVector[Double]): (Double, DenseVector[Double]) = {
         (costFunction(coef), costFunctionGradient(coef))
       }
    }

    // minimize - uses LBFGS by default
    // TODO: allow user to pass arguments here
    val optimalCoef = minimize(f, DenseVector.fill(Xbias.cols){0.0d})

    // create fitted estimator to be returned
    val trainedModel = new LogisticRegressionEstimator()
    trainedModel.w = Some(optimalCoef)

    // training score
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

  override def predict(X: DenseMatrix[Double]): DenseVector[Int] = {

    val predProbs = predictProb(X)
    // check if > 0.5 - if yes 1, else 0
    // see if nicer way to do this...
    convert(breeze.numerics.I(predProbs :>= 0.5).toDenseVector, Int)
  }

  override def predictProb(X: DenseMatrix[Double]): DenseVector[Double] = {
    // TODO: make a function to add bias, as very common operation
    val ones = DenseVector.fill(X.rows){1.0d}
    val Xbias = DenseMatrix.horzcat(new DenseMatrix(rows = X.rows, cols = 1, ones.toArray), X)

    val XCoef = Xbias(*, ::) :* _coef
    sigmoid(sum(XCoef(*, ::)))
  }

}


object LogisticRegressionEstimator {

  def apply(scoreFunc: (DenseVector[Int], DenseVector[Int]) => Double = Accuracy): LogisticRegressionEstimator =
    new LogisticRegressionEstimator(scoreFunc)

}
