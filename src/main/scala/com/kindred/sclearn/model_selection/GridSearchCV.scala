package com.kindred.sclearn.model_selection

import breeze.linalg.{DenseMatrix, DenseVector}
import com.kindred.sclearn.estimator.BaseEstimator

/*
idea: deviate from sklearn api here. a GridSearchCV has a run method, which should
return an estimator of type V

this differs from sklearn which fits a gridsearchcv object which procures the powers of the underlying
estimator

 */

// case class to encapsulate output.
// this way can remain functional, no need to update a GridSearchCV object and
// make it inherit from BaseEstimator traits etc
case class GridSearchCVResult[V](resampleResults: List[((Map[String, Any], Int), Double)],
                                 finalEstimator: V,
                                 bestScore: Double,
                                 bestParams: Map[String, Any])


class GridSearchCV[T, V <: BaseEstimator[T]](estimator: V,
                                             paramGrid: List[Map[String, Any]],
                                             scoring: (DenseVector[T], DenseVector[T]) => Double,
                                             cv: BaseCrossValidator) {


  def run(X: DenseMatrix[Double], y: DenseVector[T]): GridSearchCVResult[estimator.Y] = {

    val resampIndexStream = cv.split(X)
    val nResamples = cv.getNSplits

    // List of tuples of ((hyperparamters, resample index), score)
    val results: List[((Map[String, Any], Int), Double)] = for {
      hp <- paramGrid
      ix <- resampIndexStream.zipWithIndex

      folds = ix._1
      foldix = ix._2

      trainX = X(folds._2, ::).toDenseMatrix
      trainy = y(folds._2).toDenseVector
      testX = X(folds._1, ::).toDenseMatrix
      holdouty = y(folds._1).toDenseVector

      estimatorFold: estimator.type = estimator(hp)
      fittedEstimatorFold = estimatorFold.fit(trainX, trainy)
      holdoutPredictions = fittedEstimatorFold.predict(testX)
      modelscore = scoring(holdoutPredictions, holdouty)
    } yield ((hp, foldix),  modelscore)


    /*
    val res: List[((Map[String, Any], Int), Double)] = List(((Map("penalty" -> "l1", "C" -> 0.01), 0) , 0.8),
     ((Map("penalty" -> "l1", "C" -> 0.01), 1), 0.89),
     ((Map("penalty" -> "l1", "C" -> 0.01), 2), 0.88),
     ((Map("penalty" -> "l1", "C" -> 0.1), 0) , 0.89),
     ((Map("penalty" -> "l1", "C" -> 0.1), 1), 0.93),
     ((Map("penalty" -> "l1", "C" -> 0.1), 2), 0.91)
    )


    // continue from here...
    res.groupBy(_._1._1)

     */

    // average the results for each hyperparameter. can probably do with a groupBy
    val avgResults: Map[Map[String, Any], Double] = ???
    // which params give the best average score. can do with a sort
    val bestParams: Map[String, Any] = ???
    // what is the best average score
    val bestScore: Double = ???

    // refit the model to whole training set, using best params
    val finalEstimator: estimator.type = estimator(bestParams)
    val fittedFinalEstimator = finalEstimator.fit(X, y)

    // return the results
    GridSearchCVResult(results, fittedFinalEstimator, bestScore, bestParams)

  }

}

object GridSearchCV {

  def apply[T, V <: BaseEstimator[T]](estimator: V,
                                      paramGrid: List[Map[String, Any]],
                                      scoring: (DenseVector[T], DenseVector[T]) => Double,
                                      cv: KFold): GridSearchCV[T, V] = {
    new GridSearchCV[T, V](estimator, paramGrid, scoring, cv)
  }

}


