package fastserve.expressions

import fastserve.{FastTransformer, PlainDataset}
import org.apache.spark.sql.catalyst.expressions.{Alias, Expression, ScalaUDF}
import org.apache.spark.sql.types.{StructField, StructType}

trait Aliases extends UDFResolver {

  def applyAlias(
    curr: FastTransformer,
    schema: StructType,
    alias: Alias
  ): (FastTransformer, StructType) =
    resolveAliasChild(curr, schema, alias.name, alias.child)

  def resolveAliasChild(
    t: FastTransformer,
    schema: StructType,
    name: String,
    e: Expression
  ): (FastTransformer, StructType) = e match {
    case udf: ScalaUDF =>
      val next = t.andThen(applyUDF(name, udf, schema))
      val field = StructField(name, udf.dataType, udf.nullable)
      println(s"Alias add field: $field")
      next -> schema.add(field)
    case a: Alias =>
      resolveAliasChild(t, schema, name, a.child)
    case x => throw new NotImplementedError(s"Unexpected alias($name) child: $x, ${x.getClass}")
  }
}

object Aliases extends Aliases

