package com.kindred.sclearn.model_selection


// keep this like this- when use pipelines will do getOrElse from the map, and then cast as InstanceOf correct type
// e.g. Logistic regrssion - C is a double, so we will get C from parameter mapp and create a new logisticregrssionestimator
// where c = paramgrid.getPOrElse("C", 1.0).asInstanceOf[Double]

  object ParameterGrid {

    // perhaps models need to take a default list of tuneable parameters for each model. This should be easily extensible
    // i.e. logistic regression has by default penalty type, penalty value
    // other opt options should be passed. block user from specifiying penalty and value with opt options

    def cross(inputs: Map[String, List[Any]]) : List[Map[String, Any]] =
      inputs.foldRight(List[Map[String, Any]](Map.empty[String, Any])){(el, rest) =>
        el._2.flatMap {e =>
          rest.map {
            x => {
              x ++ Map(el._1 -> e)
            }
          }
        }
      }
  }



//
//trait Grid[T] extends Iterable[T]
//case class GridParamList[T <: AnyVal](paramList: List[T]) extends Grid[T] {
//
//  override def iterator: Iterator[T] = paramList.iterator
//
//}
//
//// This is a very difficult problem }:(
//// perhaps shapeless required to do elegantly, or some way of tracking metadata like spark rows do
//
//object GridParamList {
//
//  def apply[T](paramList: T*): GridParamList[T] = new GridParamList[T](paramList.toList)
//
//  def apply(paramList: Double*): GridParamListDouble = new GridParamListDouble(paramList.toList)
//
//  def apply(paramList: Int*): GridParamListString = new GridParamListString(paramList.toList)
//
//}
//
//class GridParamListDouble(override val paramList: List[Double]) extends GridParamList[Double](paramList)
//class GridParamListString(override val paramList: List[Int]) extends GridParamList[Int](paramList)
//
//
//
//  /*
//  limit to taking a list of params
//
//  val p = Map("penalty" -> GridParamList(List("l1", "l2")), "lambda" -> GridParamList(List(0.01, 0.1, 1.0)))
//
//  val d = Map('a' -> List(1, 2)), Map('b' -> List(5,6))
//  desired output (stream to list):
//
//  List(Map("a" -> 1, "b" -> 5), Map("a" -> 1, "b" -> 6), Map("a" -> 2, "b" -> 5), Map("a" -> 2, "b" -> 6))
//
//  def pg(l : List[Map[String, List[Int]]]): List[Map[String, Int]] = {
//    for {
//      x <-
//    }
//  }
//
//  def cross[T](inputs: List[List[T]]) : List[List[T]] =
//      inputs.foldRight(List[List[T]](Nil))((el, rest) => el.flatMap(p => rest.map(p :: _)))
//
//
//  def crossMap[A, B, C](inputs: [List[Map[A, List[B]]]) : List[Map[A, C]] =
//      inputs.foldRight(List[Map[A, C]](Nil))((el, rest) => el.flatMap(p => rest.map(p :: _)))
//
//   */
//
//object ParameterGrid extends App {
//
////  def cross[T](inputs: Map[String, List[T]]) : List[Map[String, T]] = {
////    inputs.foldRight(List[Map[String, T]](Map.empty[String, T]))((el, rest) => (p => rest.map(p :: _)))
////  }
//
//  def cross2[T](inputs: List[List[T]]) : List[List[T]] =
//    inputs.foldRight(List[List[T]](Nil)){(el, rest) =>
//      el.flatMap{p =>
//        rest.map{
//          x => p :: x
//        }
//      }
//    }
//
//  def cross[T](inputs: Map[String, List[T]]) : List[Map[String, T]] =
//    inputs.foldRight(List[Map[String, T]](Map.empty[String, T])){(el, rest) =>
//      el._2.flatMap {e =>
//        rest.map {
//          x => {
//            x ++ Map(el._1 -> e)
//          }
//        }
//      }
//    }
//
//  def cross3[T <: AnyVal](inputs: Map[String, GridParamList[T]]) : Iterable[Map[String, T]] =
//    inputs.foldRight(Iterable[Map[String, T]](Map.empty[String, T])){(el, rest) =>
//      el._2.flatMap {e =>
//        rest.map {
//          x => {
//            x ++ Map(el._1 -> e)
//          }
//        }
//      }
//    }
//
//
//
//  val p = Map("penalty" -> GridParamList(1.0, 1.0),
//    "lambda" -> GridParamList(1, 1))
//
//  p
//
//
//  print(cross3(p))
//
//
//
////  def ParameterGrid(params: List[Map[String, GridParamList]]): Stream[Map[String, GridParam]] = {
////    // need to iterate over all keys, create the products
////    ???
////  }
//
//}
