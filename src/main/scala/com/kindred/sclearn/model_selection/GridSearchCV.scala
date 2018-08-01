package com.kindred.sclearn.model_selection

import breeze.linalg.{DenseMatrix, DenseVector}
import com.kindred.sclearn.estimator.BaseEstimator


// TODO: doesnt need to be a class. can just be a function in an object.

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

  // do not make this public facing- applu method to access
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

    // averagethe results by each hyperparameter
    val avgResults: List[(Map[String, Any], Double)] = results.groupBy(_._1._1).mapValues(x => x.map(_._2).sum / x.map(_._2).length).toList
    // which params give the best average score. can do with a sort
    val bestParams: Map[String, Any] = avgResults.sortBy(_._2).tail.head._1
    // what is the best average score
    val bestScore: Double = avgResults.sortBy(_._2).tail.head._2

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


