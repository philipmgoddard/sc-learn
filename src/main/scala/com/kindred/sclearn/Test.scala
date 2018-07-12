package com.kindred.sclearn

import breeze.linalg.{DenseMatrix, DenseVector}
import com.kindred.sclearn.metrics.RegressionMetrics._
import breeze.optimize.{L1Regularization, L2Regularization}
import com.kindred.sclearn.linear_model.{LinearRegressionEstimator, LogisticRegressionEstimator}
import com.kindred.sclearn.model_selection.KFold

object Test extends App {

  //val newObs = breeze.linalg.DenseMatrix((1.0, 4.3, 5.4), (1.2, 0.4, 0.3))
  val features = DenseMatrix((1.0, 4.0, 2.0), (6.3, 8.7, 2.3), (6.6, 8.9, 2.1))
  val outcome = DenseVector(0.3, 5.0, 4.9)
  val lr_est =  LinearRegressionEstimator(scoreFunc = R2, optOptions = List(L2Regularization(0.001))).fit(features, outcome)
  val lr_est2 =  LinearRegressionEstimator(scoreFunc = R2).fit(features, outcome)



  val ypred: DenseVector[Double] = lr_est.predict(features)

  //val lr_est2 = LinearRegressionEstimator()


//  val y: DenseVector[Double] = ???
//  val ypred: DenseVector[Double] = ???

  println(lr_est.score(ypred, outcome)) // default: RMSE
  println(lr_est.score(ypred, outcome, R2)) // custom

  println(lr_est._coef)
  println(lr_est._score)
  println(lr_est)

  val HW_data = DenseMatrix((77.0, 182.0), (53.0, 161.0), (65.0, 171.0), (70.0, 175.0))
  val HW_outcome = DenseVector(1,0,1,1)

  val logistic_est =  LogisticRegressionEstimator(optOptions = List(L1Regularization(0.001))).fit(HW_data, HW_outcome)
  println(logistic_est._coef)
  println(logistic_est.predictProb(HW_data))
  println(logistic_est.predict(HW_data))
  //print(logistic_est.predictProb(HW_data))


  val kfTest = DenseMatrix((77.0, 182.0), (53.0, 161.0), (65.0, 171.0), (70.0, 175.0), (53.4, 161.2), (52.0, 12.0))

  val kf = KFold()
  println(kf.split(kfTest).toList)

  val kf2 = KFold( nSplit = 6 )
  println(kf2.split(kfTest).toList)

  val kf3 = KFold(shuffle = false )
  println(kf3.split(kfTest).toList)



  // would a foreach do the same thing?
  def printFolds(X: DenseMatrix[Double], folds: Stream[(IndexedSeq[Int], IndexedSeq[Int])]): Unit = folds match {
    case Stream.Empty => println("done")
    case x #:: xs => {
      println("")
      println("test")
      println(X(x._1, ::))
      println("train")
      println(X(x._2, ::))
      printFolds(X, xs)
    }
  }

  printFolds(kfTest, kf2.split(kfTest))
  println(kf2.getNSplits)

}


/*
 * GOAL: linear classifier and regression. Random forest for regression and classification.
 * GridSearchCV, transformer, pipeline
 *
 *
 * */
// TODO: start implementing tests
// TODO implement gridsearchCV. Think about how can hold all training metrics (CV scores, variable importance etc) within
// TODO implement feature transformers. Fit, transform, as per sklearn
// TODO implement pipelines. string together transformers, estimator to make something beautiful

// TODO think how can do gridsearchCV in parallel??
// TODO implement a couple more models... RF? Neural net?
