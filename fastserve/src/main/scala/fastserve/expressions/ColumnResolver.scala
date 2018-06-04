package fastserve.expressions


import fastserve.PlainDataset
import org.apache.spark.sql.catalyst.analysis.UnresolvedAttribute
import org.apache.spark.sql.catalyst.expressions.{AttributeReference, Cast, Expression}
import org.apache.spark.sql.types._

trait ColumnResolver {

  def resolveColumnFully(e: Expression, schema: StructType): (String, DataType, Option[LocalTransform]) = e match {
    case ref: AttributeReference => (ref.name, ref.dataType, None)
    case att: UnresolvedAttribute =>
      schema.find(_.name == att.name) match {
        case None => throw new IllegalStateException(s"Can't resolve attr: $att")
        case Some(t) => (att.name, t.dataType, None)
      }
    case cast: Cast =>
      val (name, dataType, other) = resolveColumnFully(cast.child, schema)
      val target = Casts.applyCast(name, dataType, cast.dataType)
      val transform = other match {
        case Some(f) => f.andThen(target)
        case None => target
      }
      (name, cast.dataType, Some(transform))
    case x => throw new IllegalArgumentException(s"Unexpected expression in column resolution: $x")
  }

  def resolveColumn(e: Expression, schema: StructType): (String, Option[LocalTransform]) = {
    val out = resolveColumnFully(e, schema)
    out._1 -> out._3
  }

}


