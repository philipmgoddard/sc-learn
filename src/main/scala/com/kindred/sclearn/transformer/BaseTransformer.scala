package com.kindred.sclearn.transformer

import breeze.linalg.{DenseMatrix}

trait BaseTransformer {

  def transform(DenseVector: DenseMatrix[Double]): DenseMatrix[Double]

}
