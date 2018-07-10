package com.kindred.sclearn.model_selection

import breeze.linalg.DenseMatrix

/*
 methods are split, iter_test_indices

 */
trait BaseCrossValidator {

  def split(X: DenseMatrix[Double]): Stream[(Seq[Int], Seq[Int])]

  def get_n_splits(): Int

}
