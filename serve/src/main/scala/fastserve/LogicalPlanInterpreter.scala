package fastserve

import org.apache.spark.ml.Transformer
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.catalyst.analysis.UnresolvedAttribute
import org.apache.spark.sql.catalyst.expressions.{Alias, AttributeReference, ScalaUDF}
import org.apache.spark.sql.catalyst.plans.logical.{AnalysisBarrier, LogicalPlan, Project, ReturnAnswer}
import org.apache.spark.sql.execution.LogicalRDD

import scala.annotation.switch

object LogicalPlanInterpreter {

  def mkFastTransformer(plan: LogicalPlan): FastTransformer = mkFastTransformer(ConstFastTransformer, plan)

  def mkFastTransformer(curr: FastTransformer, plan: LogicalPlan): FastTransformer = plan match {
    case barr: AnalysisBarrier => mkFastTransformer(curr, barr.child)
    case lRdd: LogicalRDD => curr
    case Project(exprs, child) =>
      val trasformations = exprs.collect{
        case alias @ Alias(child, name) =>
          child match {
            case s @ ScalaUDF(func, _, children, _, name, nullable, determ) =>
              val inputNames = children.map{
                case ref: AttributeReference => ref.name
                case att: UnresolvedAttribute => att.name
              }

              val f = (d: PlainDataset) => {
                val targets = inputNames.map(n => d.columns(d.columnsId(n)))
                val values = (targets.size: @switch) match {
                  case 0 => throw new RuntimeException("WTF? zero input udf")
                  case 1 => targets.head.items.map(i => Util.callUDF1(func, i))
                  case x =>
                    (0 until d.size).map(i => {
                      val in = targets.map(c => c.items(i))
                      Util.callUDF(func, in)
                    })
                }
                d.addColumn(Column(alias.name, values))
              }
              Some(f)
            case x =>
              println(s"Alias $alias whith $x")
              None
          }
        case other =>
          println(s"Ignore projection expr: $other")
          None
      }
      val next = trasformations.flatten.foldLeft(curr){case (t, f) => t.compose(f)}
      mkFastTransformer(next, child)
  }

  def fromTransformer(t: Transformer, sample: Dataset[Row]): FastTransformer = {
    val plan = t.transform(sample).queryExecution.logical
    mkFastTransformer(plan)
  }
}
