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


/*
class LR(c: Double = LR.c, penalty: String = LR.penalty) {
  println(c)
}

object LR {
  val c = 1.1
  val penalty = "l1"
}

object Test extends App {
  val params: Map[String, Any] = Map("penalty" -> 1)
  new LR(c = params.getOrElse("c", LR.c).asInstanceOf[Double],
    penalty = params.getOrElse("penalty", LR.penalty).asInstanceOf[String])
}
 */