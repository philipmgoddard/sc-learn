package com.kindred.sclearn.model_selection
import breeze.linalg.{DenseMatrix, DenseVector}
import com.kindred.sclearn.estimator.BaseModel


object SearchCV {




  // case class to encapsulate output.
  // this way can remain functional, no need to update a GridSearchCV object and
  // make it inherit from BaseEstimator traits etc
  case class SearchCVResult(resampleResults: List[((Map[String, Any], Int), Double)],
                            finalEstimator: BaseModel,
                            bestScore: Double,
                            bestParams: Map[String, Any])


  // GridSearchCV
  // TODO: is there a better way to define estimator as a subtype of Baseestimator???
  def GridSearchCV(estimator: BaseModel,
                   paramGrid: List[Map[String, Any]],
                   scoring: (DenseVector[Double], DenseVector[Double]) => Double,
                   biggerIsBetter: Boolean,
                   cv: BaseCrossValidator)(X: DenseMatrix[Double], y: Option[DenseVector[Double]]): SearchCVResult = {

    val yVal = y.get
    val resampIndexStream = cv.split(X)
    val nResamples = cv.getNSplits


    // List of tuples of ((hyperparamters, resample index), score)
    val results: List[((Map[String, Any], Int), Double)] = for {
      hp <- paramGrid
      ix <- resampIndexStream.zipWithIndex

      folds = ix._1
      foldix = ix._2

      trainX = X(folds._2, ::).toDenseMatrix
      trainy = yVal(folds._2).toDenseVector
      holdoutX = X(folds._1, ::).toDenseMatrix
      holdouty = yVal(folds._1).toDenseVector

      // do not want this run method
      estimatorFold = estimator.run(hp)

      fittedEstimatorFold = estimatorFold.fit(trainX, Option(trainy))
      holdoutPredictions = fittedEstimatorFold.predict(holdoutX)
      modelscore = scoring(holdoutPredictions, holdouty)
    } yield ((hp, foldix), modelscore)

    // average the results by each hyperparameter
    val avgResults: List[(Map[String, Any], Double)] = results
      .groupBy(_._1._1)
      .mapValues(x => x.map(_._2).sum / x.map(_._2).length)
      .toList

    val bestResult = if (biggerIsBetter) avgResults.maxBy(_._2) else avgResults.minBy(_._2)
    val bestParams: Map[String, Any] = bestResult._1
    val bestScore: Double = bestResult._2

    // refit the model to whole training set, using best params
    val fittedFinalEstimator: BaseModel= estimator.run(bestParams).fit(X, y)

    // return the results
    SearchCVResult(results, fittedFinalEstimator, bestScore, bestParams)
  }
}
