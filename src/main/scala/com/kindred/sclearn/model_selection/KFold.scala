package com.kindred.sclearn.model_selection

import breeze.linalg.DenseMatrix


class KFold(nSplits: Int, shuffle: Boolean, randomState: Int) extends BaseCrossValidator {

  // Set the seed on initilisation
  scala.util.Random.setSeed(seed = randomState)


  // generate a stream of (train_ix, test_ix) for slicing a breeze dense matrix
  override def split(X: DenseMatrix[Double]): Stream[(IndexedSeq[Int], IndexedSeq[Int])] = {

    if (X.rows < nSplits) throw new IllegalArgumentException("Cant have more splits than observations")

    val nObs = X.rows
    val obsPerFold = nObs / nSplits
    val ix = if (shuffle) {
      scala.util.Random.shuffle((0 until nObs).toVector)
    } else {
      (0 until nObs).toVector
    }

    def _split(ixLo: Int, ixHi: Int, currSol: Stream[(IndexedSeq[Int], IndexedSeq[Int])]): Stream[(IndexedSeq[Int], IndexedSeq[Int])] = {
      val posLo = scala.math.min(nObs, ixLo)
      val posHi = scala.math.min(nObs, ixHi)
      if (posLo == nObs) currSol
      else {
        val test = ix.slice(ixLo, posHi)
        val train = ix.take(posLo) ++ ix.drop(posLo + obsPerFold)
        _split(ixHi, ixHi + obsPerFold, currSol ++ Stream((test, train)))
      }
    }

    _split(ixLo = 0, ixHi = obsPerFold, currSol = Stream.empty)
  }


  // getter for number of splits.
  override def getNSplits: Int = nSplits

}

object KFold {

  def apply(nSplit: Int = 3, shuffle: Boolean = true, randomState: Int = 1234): KFold = {
    new KFold(nSplit, shuffle, randomState)
  }

}