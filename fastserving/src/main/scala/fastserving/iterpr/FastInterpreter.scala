package fastserving.iterpr

import fastserving.{ConstFastTransformer, FastTransformer}
import org.apache.spark.ml.Transformer
import org.apache.spark.sql.catalyst.analysis.{UnresolvedAttribute, UnresolvedStar}
import org.apache.spark.sql.catalyst.expressions.{Alias, AttributeReference}
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.catalyst.trees.TreeNode
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Dataset, Row}

import scala.annotation.tailrec

object FastInterpreter extends UDFResolver {

  def mkFastTransformer(plan: LogicalPlan, schema: StructType): FastTransformer =
    interp(plan, ConstFastTransformer, schema)

  private def interp(root: TreeNode[_], curr: FastTransformer, schema: StructType): FastTransformer = {
    val projects = selectProjections(root, List.empty)
    val (tF, sF) = projects.foldLeft((curr, schema)) {
      case ((t, s), pr) =>
        pr.expressions.foldLeft((t, s)){
          case ((t, schema), _: AttributeReference) => (t, schema)
          case ((t, schema), _: UnresolvedStar) => (t, schema)
          case ((t, schema), al @ Alias(child, name)) => Aliases.applyAlias(t, schema, al)
          case ((_, _), x) => throw new NotImplementedError(s"Unexpected expression in project($pr): $x, ${x.getClass}")
        }
    }
    tF
  }

  /**
    * Reverse logical plan and select only important projections
    */
  @tailrec
  private def selectProjections(node: Any, acc: List[Project]): List[Project] = node match {
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
    case _: LeafNode => acc
    case x => throw new NotImplementedError(s"Unexpected expression $x: ${x.getClass}")
  }

  def fromTransformer(t: Transformer, sample: Dataset[Row]): FastTransformer = {
    val plan = t.transform(sample).queryExecution.logical
    mkFastTransformer(plan, sample.schema)
  }

}
