package com.kindred.sclearn

import breeze.linalg.{DenseMatrix, DenseVector}
import com.kindred.sclearn.metrics.RegressionMetrics._
import breeze.optimize.{L1Regularization, L2Regularization}
import com.kindred.sclearn.estimator.BaseEstimator
import com.kindred.sclearn.linear_model.{LinearRegressionEstimator, LogisticRegressionEstimator}
import com.kindred.sclearn.metrics.{ClassificationMetrics, RegressionMetrics}
import com.kindred.sclearn.model_selection._

object Test extends App {

  //val newObs = breeze.linalg.DenseMatrix((1.0, 4.3, 5.4), (1.2, 0.4, 0.3))
  val features = DenseMatrix((1.0, 4.0, 2.0), (6.3, 8.7, 2.3), (6.6, 8.9, 2.1))
  val outcome = DenseVector(0.3, 5.0, 4.9)
//  val lr_est =  LinearRegressionEstimator(scoreFunc = R2, optOptions = List(L2Regularization(0.001))).fit(features, outcome)
  val lr_est =  LinearRegressionEstimator(penalty = "l2", C = 10.0).fit(features, outcome)

  val lr_est2 =  LinearRegressionEstimator(Map("penalty" -> "l1", "C" -> 10.0)).fit(features, outcome)


  println(lr_est)
  println("hello")
  println(features)

  val ypred: DenseVector[Double] = lr_est.predict(features)

  println(ypred)

  //val lr_est2 = LinearRegressionEstimator()


//  val y: DenseVector[Double] = ???
//  val ypred: DenseVector[Double] = ???

  println(lr_est.score(ypred, outcome)) // default: R2
  println(lr_est.score(ypred, outcome, RMSE)) // custom

  println(lr_est.coef_)
//  println(lr_est.optOptions)
//
//  val HW_data = DenseMatrix((77.0, 182.0), (53.0, 161.0), (65.0, 171.0), (70.0, 175.0))
//  val HW_outcome = DenseVector(1,0,1,1)
//
//  val logistic_est =  LogisticRegressionEstimator(optOptions = List(L1Regularization(0.001))).fit(HW_data, HW_outcome)
//  println(logistic_est._coef)
//  println(logistic_est.predictProb(HW_data))
//  println(logistic_est.predict(HW_data))
//  //print(logistic_est.predictProb(HW_data))
//
//
  val kfTest = DenseMatrix((77.0, 182.0), (53.0, 161.0), (65.0, 171.0), (70.0, 175.0), (53.4, 161.2), (52.0, 12.0))

  val x = 1

  val kf = KFold()
  println(kf.split(kfTest).toList)

  val kf2 = KFold( nSplit = 6 )
  println(kf2.split(kfTest).toList)

  val kf3 = KFold(shuffle = false )
  println(kf3.split(kfTest).toList)

  val tmp = kf3.split(kfTest)

  val tt = kfTest(tmp(0)._1, ::).toDenseMatrix


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

  kf2.split(kfTest).foreach{case (x,y) =>
    println("")
    println("test")
    println(kfTest(x, ::))
    println("train")
    println(kfTest(y, ::))
  }


  val parGrid = ParamGrid.cross(Map("penalty" -> List("l1", "l2"), "C" -> List(0.01, 0.1, 1.0)))


  val gs2 = SearchCV.GridSearchCV(estimator =  lr_est , paramGrid = parGrid, scoring=  RMSE , cv=  KFold(nSplit = 2))(features, outcome)

//  println(res)
  println(gs2)
  println(gs2.bestScore, gs2.bestParams)
  val avgres = gs2.resampleResults
    .groupBy(_._1._1)
    .mapValues(x => x.map(_._2).sum / x.map(_._2).length)
    .toList

  println("")
  println(avgres.sortBy(_._2))//.tail)//.head._1)
  println(avgres.sortBy(_._2).tail)//.head._1)
  println(avgres.sortBy(_._2).tail.head)//._1)
  println(avgres.sortBy(_._2).tail.head._1)

  val xx = gs2.resampleResults.sortBy(_._2).tail.head._1


//  printFolds(kfTest, kf2.split(kfTest))
//  println(kf2.getNSplits)


//  val e  = LinearRegressionEstimator()
//
//  val yPred: DenseVector[Double] = ???
//
//  val scoring = RegressionMetrics.RMSE _
//
//  val ee = GridSearchCV(e, scoring)

//  ee

}
//
//
///*
// * GOAL: linear classifier and regression. Random forest for regression and classification.
// * GridSearchCV, transformer, pipeline
// *
// *
// * */
//// TODO: start implementing tests
//// TODO implement gridsearchCV. Think about how can hold all training metrics (CV scores, variable importance etc) within
//// TODO implement feature transformers. Fit, transform, as per sklearn
//// TODO implement pipelines. string together transformers, estimator to make something beautiful
//   TODO: implicits for scoring ordering- to say if small or big is better
//
//// TODO think how can do gridsearchCV in parallel??
//// TODO implement a couple more models... RF? Neural net?

//class GridSearchCV[T, V <: BaseEstimator[T]](estimator: V) {
//  def run(): V = ???
//}
//
//object GridSearchCV {
//
//  def apply[T, V <: BaseEstimator[T]](estimator: V, scoring: (DenseVector[T], DenseVector[T]) => Double): V =
//    new GridSearchCV[T, V](estimator).run()
//
//}
