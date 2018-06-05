package fastserve.expressions


import fastserve.PlainDataset
import org.apache.spark.sql.catalyst.analysis.UnresolvedAttribute
import org.apache.spark.sql.catalyst.expressions.{Alias, AttributeReference, Cast, Expression, ScalaUDF}
import org.apache.spark.sql.types._

trait ColumnResolver  {

  def resolveColumn(e: Expression, schema: StructType): (String, StructType, Option[LocalTransform]) = e match {
    case ref: AttributeReference => //(ref.name, ref.dataType, None)
      val field = schema.find(_.name == ref.name).getOrElse(StructField(ref.name, ref.dataType, ref.nullable, ref.metadata))
      (ref.name, schema.add(field), None)
    case att: UnresolvedAttribute =>
      schema.find(_.name == att.name) match {
        case None => throw new IllegalStateException(s"Can't resolve attr: $att")
        case Some(_) => (att.name, schema, None)
      }
    case cast: Cast =>
      val (name, patched, other) = resolveColumn(cast.child, schema)
      val field = patched.find(_.name == name) match {
        case None => throw new IllegalStateException(s"Can't resolve cast: $cast")
        case Some(f) => f
      }
      val target = Casts.applyCast(name, field.dataType, cast.dataType)
      val transform = other match {
        case Some(f) => f.andThen(target)
        case None => target
      }
      val casted = StructType(
        patched.map(f => {
          if (f.name == name) StructField(name, cast.dataType, f.nullable)
          else f
        })
      )
      (name, casted, Some(transform))
    case x => throw new IllegalArgumentException(s"Unexpected expression in column resolution: $x")
  }

  def resolveColumns(exprs: Seq[Expression], schema: StructType): (Seq[String], StructType, LocalTransform) = {
    exprs.foldLeft((Seq.empty[String], schema, identity[PlainDataset] _)){
      case ((names, sch, tr), e) =>
        val (name, patched, maybeT) = resolveColumn(e, sch)
        maybeT match {
          case Some(f) => (names :+ name, patched, tr.compose(f))
          case None => (names :+ name, patched, tr)
        }
    }
  }

}


