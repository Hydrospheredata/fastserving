package fastserving.interp

import fastserving.interp.compat.LogicalPlanCompat
import fastserving.{ConstFastTransformer, FastTransformer}
import org.apache.spark.ml.Transformer
import org.apache.spark.sql.catalyst.analysis.UnresolvedStar
import org.apache.spark.sql.catalyst.expressions.{Alias, AttributeReference}
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Dataset, Row}


object FastInterpreter extends UDFResolver {

  def mkFastTransformer(plan: LogicalPlan, schema: StructType): FastTransformer =
    interp(plan, ConstFastTransformer, schema)

  private def interp(plan: LogicalPlan, curr: FastTransformer, schema: StructType): FastTransformer = {
    val projects = LogicalPlanCompat.selectProjections(plan, List.empty)
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

  def fromTransformer(t: Transformer, sample: Dataset[Row]): FastTransformer = {
    val plan = t.transform(sample).queryExecution.logical
    mkFastTransformer(plan, sample.schema)
  }

}
