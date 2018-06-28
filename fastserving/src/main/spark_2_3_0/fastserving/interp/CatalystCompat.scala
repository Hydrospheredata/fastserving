package fastserving.interp

import org.apache.spark.sql.catalyst.analysis.{UnresolvedAttribute, UnresolvedStar}
import org.apache.spark.sql.catalyst.expressions.AttributeReference
import org.apache.spark.sql.catalyst.plans.logical.{LogicalPlan, Project}

import scala.annotation.tailrec

object CatalystCompat {

  @tailrec
  def selectProjections(node: Any, acc: List[Project]): List[Project] = node match {
    case barr: AnalysisBarrier => selectProjections(barr.child, acc)
    case pr: Project =>
      pr.child match {
        case _ =>
          val ignore = pr.expressions.forall{
            case _: AttributeReference => true
            case _: UnresolvedStar => true
            case _: UnresolvedAttribute => true
            case _ => false
          }
          val next = if (ignore)  acc else pr :: acc
          selectProjections(pr.child, next)
      }
    case _: LogicalPlan => acc
    case x => throw new NotImplementedError(s"Unexpected expression $x: ${x.getClass}")
  }

}
