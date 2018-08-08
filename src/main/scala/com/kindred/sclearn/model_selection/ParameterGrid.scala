package com.kindred.sclearn.model_selection

  object ParameterGrid {

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
