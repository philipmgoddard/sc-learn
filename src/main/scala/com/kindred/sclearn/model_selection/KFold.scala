package com.kindred.sclearn.model_selection

import breeze.linalg.DenseMatrix


/*
sk-learn returns a generator with indices for slicing train/test from input.

return a stream of tuples of indexedSeq for each fold
i guess a stream is the right thing to return?

 */



class KFold(n_splits: Int, shuffle: Boolean, random_state: Int) extends BaseCrossValidator {

  // this has a side effect.
  scala.util.Random.setSeed(seed = random_state)

  val ix = 0 until X.rows

  val indexToSplit = if (shuffle) {
    shuffleIndex(ix)

  }


  def split(X: DenseMatrix[Double]):Stream[(Seq[Int], Seq[Int])] ={

    n_splits
    ???

    // build up the stream for the number of splits
  }



  def shuffleIndex(ix: Seq[Int]): Seq[Int] = {
    ???
  }

}
