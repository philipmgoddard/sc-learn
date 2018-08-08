package com.kindred.sclearn.model_selection

import breeze.linalg.{DenseMatrix, DenseVector}
import com.kindred.sclearn.estimator.BaseEstimator

import scala.reflect.ClassTag

object SearchCV {




  // case class to encapsulate output.
  case class SearchCVResult[V](resampleResults: List[((Map[String, Any], Int), Double)],
                               finalEstimator: V,
                               bestScore: Double,
                               bestParams: Map[String, Any])

  // GridSearchCV
  // curry the function rather than just having a run method for fun
  def GridSearchCV[T: ClassTag](estimator: BaseEstimator[T],
                                paramGrid: List[Map[String, Any]],
                                scoring: (DenseVector[T], DenseVector[T]) => Double,
                                cv: BaseCrossValidator)(X: DenseMatrix[Double], y: DenseVector[T]): SearchCVResult[BaseEstimator[T]] = {

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
      holdoutX = X(folds._1, ::).toDenseMatrix
      holdouty = y(folds._1).toDenseVector

      estimatorFold: BaseEstimator[T] = estimator.run(hp)

      fittedEstimatorFold = estimatorFold.fit(trainX, trainy)
      holdoutPredictions = fittedEstimatorFold.predict(holdoutX)
      modelscore = scoring(holdoutPredictions, holdouty)
    } yield ((hp, foldix), modelscore)

    // average the results by each hyperparameter
    val avgResults: List[(Map[String, Any], Double)] = results
      .groupBy(_._1._1)
      .mapValues(x => x.map(_._2).sum / x.map(_._2).length)
      .toList

    // which params give the best average score. can do with a sort
    // TODO ask tamas about implicits for this. not all metrics are better when smaller- most better when bigger
    val bestResult = avgResults.minBy(_._2)
    val bestParams: Map[String, Any] = bestResult._1
    val bestScore: Double = bestResult._2

    // refit the model to whole training set, using best params
    val fittedFinalEstimator: BaseEstimator[T] = estimator.run(bestParams).fit(X, y)

    // return the results
    SearchCVResult(results, fittedFinalEstimator, bestScore, bestParams)
  }
}
