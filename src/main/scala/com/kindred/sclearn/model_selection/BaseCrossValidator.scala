package com.kindred.sclearn.model_selection

import breeze.linalg.DenseMatrix

// TODO check hierarchy. any actual advantage of using indexedSeq vs Seq?

trait BaseCrossValidator {

  def split(X: DenseMatrix[Double]): Stream[(IndexedSeq[Int], IndexedSeq[Int])]

  def getNSplits: Int

}
