package fastserve

import fastserve.expressions.{Aliases, Casts, UDFResolver}
import org.apache.spark.ml.Transformer
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.catalyst.analysis.UnresolvedAttribute
import org.apache.spark.sql.catalyst.expressions.{Alias, AttributeReference, Cast, Expression, NamedExpression, ScalaUDF}
import org.apache.spark.sql.catalyst.plans.logical.{AnalysisBarrier, LogicalPlan, Project, ReturnAnswer}
import org.apache.spark.sql.execution.LogicalRDD
import org.apache.spark.sql.types.{StructField, StructType}

import scala.annotation.switch

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
        case (_, x) => throw new NotImplementedError(s"Unexpected expression in project(${pr}): ${x.getClass}")
      }
      mkFastTransformer(nextT, child, nextSchema)
//      val trasformations = exprs.collect{
//        case alias @ Alias(child, name) =>
//          child match {
//            case udf: ScalaUDF => Some(applyUDF(alias.name, udf, schema))
//              val inputNames = children.map {
//                case ref: AttributeReference => ref.name
//                case att: UnresolvedAttribute => att.name
//                case cast: Cast => Casts.mkFunction(cast)
//                case x => throw new IllegalArgumentException(s"Unexpected alias child: $x")
//              }
//
//              val f = (d: PlainDataset) => {
//                val targets = inputNames.map(n => d.columns(d.columnsId(n)))
//                val values = (targets.size: @switch) match {
//                  case 0 => throw new RuntimeException("zero input udf")
//                  case 1 => targets.head.items.map(i => Util.callUDF1(func, i))
//                  case x =>
//                    (0 until d.size).map(i => {
//                      val in = targets.map(c => c.items(i))
//                      Util.callUDF(func, in)
//                    })
//                }
//                d.addColumn(Column(alias.name, values))
//              }
//              Some(f)
//            case _ => None
//          }
//        case _ => None
//      }
//      val next = trasformations.flatten.foldLeft(curr){case (t, f) => t.compose(f)}
//      mkFastTransformer(next, child, schema)
  }

  def fromTransformer(t: Transformer, sample: Dataset[Row]): FastTransformer = {
    val plan = t.transform(sample).queryExecution.logical
    mkFastTransformer(plan, sample.schema)
  }
}
