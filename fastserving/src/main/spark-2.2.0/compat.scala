package fastserving.interp.compat

import fastserving.interp.{ArgAccessor, ArgResolver, Casts}
import org.apache.spark.sql.catalyst.analysis.{UnresolvedAttribute, UnresolvedStar}
import org.apache.spark.sql.catalyst.expressions.{Alias, Attribute, AttributeReference, Cast, CreateNamedStruct, Expression}
import org.apache.spark.sql.catalyst.plans.logical.{LogicalPlan, Project}
import org.apache.spark.sql.types.{DataType, StructType}

import scala.annotation.tailrec

object LogicalPlanCompat {

  @tailrec
  def selectProjections(node: Any, acc: List[Project]): List[Project] = node match {
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

object UDFChildrenCompat {

  def resolveAtt(att: Attribute, schema: StructType): (ArgResolver, DataType) = att match {
    case ref: AttributeReference => ArgResolver.columnRef(ref.name, ref.dataType) -> ref.dataType
    case att: UnresolvedAttribute =>
      schema.find(_.name == att.name) match {
        case None => throw new IllegalStateException(s"Can't resolve attr: $att")
        case Some(f) => ArgResolver.columnRef(att.name, f.dataType) -> f.dataType
      }
  }

  def resolveCast(cast: Cast, schema: StructType): (ArgResolver, DataType) = {
    val (underlying, dt) =  cast.child match {
      case c: Cast => resolveCast(c, schema)
      case att: Attribute => resolveAtt(att, schema)
    }
    if (cast.dataType == dt) {
      underlying -> dt
    } else {
      underlying.map(Casts.castFunction(dt, cast.dataType)) -> cast.dataType
    }
  }

  def resolveAlias(alias: Alias, schema: StructType): (ArgResolver, DataType) = {
    alias.child match {
      case att: Attribute => resolveAtt(att, schema)
      case cast: Cast => resolveCast(cast, schema)
      case x => throw new IllegalArgumentException(s"Unexpected expression in udf alias resolution: $x, ${x.getClass}")
    }
  }

  def lowResolve(exp: Expression, schema: StructType): (ArgResolver, DataType) = exp match {
    case att: Attribute => resolveAtt(att, schema)
    case cast: Cast => resolveCast(cast, schema)
    case alias: Alias => resolveAlias(alias, schema)
    case x => throw new IllegalArgumentException(s"Unexpected expression in column resolution: $x, ${x.getClass}")
  }

  def resolver(expr: Expression, schema: StructType): ArgResolver = {
    def resolveStructChildren(ch: Seq[Expression]): ArgResolver = {
      val out = ch.map(v => lowResolve(v, schema))
      val resolver = ArgResolver(ds => {
        val x = out.map({case (r, dt) => r.mkAccessor(ds)})
        ArgAccessor.asRow(x)
      })
      resolver
    }

    expr match {
      case att: Attribute => resolveAtt(att, schema)._1
      case cast: Cast => resolveCast(cast, schema)._1
      case cns: CreateNamedStruct => resolveStructChildren(cns.children.grouped(2).map(_.apply(1)).toSeq)
      case x => throw new IllegalArgumentException(s"Unexpected expression in column resolution: $x, ${x.getClass}")
    }
  }
}
