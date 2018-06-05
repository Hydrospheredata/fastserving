package fastserve

import fastserve.expressions.{Aliases, UDFResolver}
import org.apache.spark.ml.Transformer
import org.apache.spark.sql.catalyst.analysis.UnresolvedStar
import org.apache.spark.sql.catalyst.expressions.{Alias, AttributeReference}
import org.apache.spark.sql.catalyst.plans.logical.{AnalysisBarrier, LogicalPlan, Project}
import org.apache.spark.sql.catalyst.trees.TreeNode
import org.apache.spark.sql.execution.LogicalRDD
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{Dataset, Row}

import scala.annotation.tailrec

object FastInterpreter extends UDFResolver {

  def mkFastTransformer(plan: LogicalPlan, schema: StructType): FastTransformer =
    interp(plan, ConstFastTransformer, schema)

//  def mkFastTransformer(curr: FastTransformer, plan: LogicalPlan, schema: StructType): FastTransformer = plan match {
//    case barr: AnalysisBarrier => mkFastTransformer(curr, barr.child, schema)
//    case _: LogicalRDD => curr
//    case pr @ Project(exprs, child) =>
//      println(s"Project: $pr")
//      val (nextT, nextSchema) = exprs.foldLeft((curr, schema)){
//        case ((t, schema), att: AttributeReference) =>
//          val field = StructField(att.name, att.dataType, att.nullable, att.metadata)
//          (t, schema.add(field))
//        case ((t, schema), al @ Alias(child, name)) => Aliases.applyAlias(t, schema, al)
//        case ((t, schema), ustar: UnresolvedStar) => (t, schema)
//        case (_, x) => throw new NotImplementedError(s"Unexpected expression in project(${pr}): ${x.getClass}")
//      }
//      mkFastTransformer(nextT, child, nextSchema)
//  }



  private def interp(root: TreeNode[_], curr: FastTransformer, schema: StructType): FastTransformer = {
    val reversed = reversePlan(root, List.empty)
    println("Size:" + reversed.size)
    reversed.foreach(x => println(s"Node: $x"))
    println("Schema:" + schema)


    val (tF, sF) = reversed.foldLeft((curr, schema)) {
      case ((t, s), barr: AnalysisBarrier) => (t, s)
      case ((t, s), rdd: LogicalRDD) => (t, s)
      case ((t, s), pr @ Project(exprs, _)) =>
        exprs.foldLeft((t, s)){
          case ((t, schema), att: AttributeReference) =>
            val field = StructField(att.name, att.dataType, att.nullable, att.metadata)
            (t, schema.add(field))
          case ((t, schema), al @ Alias(child, name)) => Aliases.applyAlias(t, schema, al)
          case ((t, schema), ustar: UnresolvedStar) => (t, schema)
          case ((_, _), x) => throw new NotImplementedError(s"Unexpected expression in project(${pr}): $x, ${x.getClass}")
        }
      case ((_, _), x)=> throw new NotImplementedError(s"Unexpected expression $x: ${x.getClass}")

    }
    tF
  }

  @tailrec
  private def reversePlan(node: Any, acc: List[Any]): List[Any] = node match {
    case barr: AnalysisBarrier => reversePlan(barr.child, barr :: acc)
    case pr: Project => reversePlan(pr.child, pr :: acc)
    case _: LogicalRDD => acc
    case x => throw new NotImplementedError(s"Unexpected expression $x: ${x.getClass}")
  }


  def fromTransformer(t: Transformer, sample: Dataset[Row]): FastTransformer = {
    val plan = t.transform(sample).queryExecution.logical
    println(plan)
    mkFastTransformer(plan, sample.schema)
  }
}
