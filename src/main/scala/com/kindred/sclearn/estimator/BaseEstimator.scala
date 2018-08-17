package com.kindred.sclearn.estimator

import breeze.linalg.{DenseMatrix, DenseVector}


// one possibility- base estimator is no longer types. everything must be on Double

trait BaseEstimator[T]{


  // protected method for use only in gridSearch. This is NOT part of public API
  // used to allow generic typing on the class
  // TODO: is run the best name for this? not really sure it describes what it does
  // literally just here because cannot have apply at this level
  protected[kindred] def run(paramMap: Map[String, Any]): BaseEstimator[T]

}
