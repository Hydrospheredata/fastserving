package fastserve

import fastserve.expressions.{Aliases, UDFResolver}
import org.apache.spark.ml.Transformer
import org.apache.spark.sql.catalyst.analysis.UnresolvedStar
import org.apache.spark.sql.catalyst.expressions.{Alias, AttributeReference}
import org.apache.spark.sql.catalyst.plans.logical.{AnalysisBarrier, LogicalPlan, Project}
import org.apache.spark.sql.execution.LogicalRDD
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{Dataset, Row}

object FastInterpreter extends UDFResolver {

  def mkFastTransformer(plan: LogicalPlan, schema: StructType): FastTransformer =
    mkFastTransformer(ConstFastTransformer, plan, schema)

  def mkFastTransformer(curr: FastTransformer, plan: LogicalPlan, schema: StructType): FastTransformer = plan match {
    case barr: AnalysisBarrier => mkFastTransformer(curr, barr.child, schema)
    case _: LogicalRDD => curr
    case pr @ Project(exprs, child) =>
      val (nextT, nextSchema) = exprs.foldLeft((curr, schema)){
        case ((t, schema), att: AttributeReference) =>
          val field = StructField(att.name, att.dataType, att.nullable, att.metadata)
          (t, schema.add(field))
        case ((t, schema), al @ Alias(child, name)) => Aliases.applyAlias(t, schema, al)
        case ((t, schema), ustar: UnresolvedStar) => (t, schema)
        case (_, x) => throw new NotImplementedError(s"Unexpected expression in project(${pr}): ${x.getClass}")
      }
      mkFastTransformer(nextT, child, nextSchema)
  }

  def fromTransformer(t: Transformer, sample: Dataset[Row]): FastTransformer = {
    val plan = t.transform(sample).queryExecution.logical
    mkFastTransformer(plan, sample.schema)
  }
}
