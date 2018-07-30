package com.kindred.sclearn.model_selection

import breeze.linalg.DenseMatrix

class KFold(nSplits: Int, shuffle: Boolean, randomState: Int) extends BaseCrossValidator {

  // Set the seed on initilisation
  scala.util.Random.setSeed(seed = randomState)

  // generate a stream of (test_ix, train_ix) for slicing a breeze dense matrix
  override def split(X: DenseMatrix[Double]): Stream[(IndexedSeq[Int], IndexedSeq[Int])] = {

    if (X.rows < nSplits) throw new IllegalArgumentException("Cant have more splits than observations")

    val nObs = X.rows
    val obsPerFold = nObs / nSplits
    val ix = if (shuffle) {
      scala.util.Random.shuffle((0 until nObs).toVector)
    } else {
      (0 until nObs).toVector
    }

    val splittingPoints = 0 to nObs by obsPerFold.max(1)
    val loHi = splittingPoints.zip(splittingPoints.tail)

    {
      loHi map (
        x => {
          val test = ix.slice(x._1, x._2)
          val train = ix.take(x._1) ++ ix.drop(x._1 + obsPerFold)
          (test, train)
          })
    }.toStream

  }

  // getter for number of splits.
  override def getNSplits: Int = nSplits

}

object KFold {

  def apply(nSplit: Int = 3, shuffle: Boolean = true, randomState: Int = 1234): KFold = {
    new KFold(nSplit, shuffle, randomState)
  }

}