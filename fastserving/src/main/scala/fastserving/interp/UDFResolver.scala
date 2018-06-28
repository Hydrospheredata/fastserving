package fastserving.iterpr

import fastserving.{Column, PlainDataset}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.catalyst.analysis.UnresolvedAttribute
import org.apache.spark.sql.catalyst.expressions.{Alias, Attribute, AttributeReference, Cast, CreateNamedStruct, CreateStruct, Expression, GenericRow, ScalaUDF}
import org.apache.spark.sql.types.{DataType, StructType}

abstract class ArgAccessor { self =>
  def get(i: Int): Any
  final def map(f: Any => Any): ArgAccessor = new ArgAccessor {
    override def get(i: Int): Any = f(self.get(i))
  }
}

object ArgAccessor {

  val VectorUDTType = ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType

  def apply(f: Int => Any): ArgAccessor = new ArgAccessor {
    override def get(i: Int): Any = f(i)
  }

  def directColumn(c: Column): ArgAccessor = ArgAccessor(i => c.items(i))
  def copyVector(c: Column): ArgAccessor = directColumn(c).map(_.asInstanceOf[Vector].copy)

  def directByType(c: Column, dt: DataType): ArgAccessor = dt match {
    case VectorUDTType => copyVector(c)
    case _ => directColumn(c)
  }

  def asRow(accs: Seq[ArgAccessor]): ArgAccessor = {
    new ArgAccessor {
      val length = accs.length
      override def get(i: Int): Any =  {
        val arrInput = new Array[Any](length)
        for { y <- 0 until length }
          arrInput(y) = accs(y).get(i)
        new GenericRow(arrInput)
      }
    }
  }
}


trait ArgResolver { self =>
  def mkAccessor(ds: PlainDataset): ArgAccessor
  final def map(f: Any => Any): ArgResolver = new ArgResolver {
    override def mkAccessor(ds: PlainDataset): ArgAccessor = self.mkAccessor(ds).map(f)
  }
}

object ArgResolver {

  def apply(f: PlainDataset => ArgAccessor): ArgResolver = new ArgResolver {
    override def mkAccessor(ds: PlainDataset): ArgAccessor = f(ds)
  }

  def columnRef(name: String, dt: DataType): ArgResolver = {
    ArgResolver(ds => ArgAccessor.directByType(ds.columnByName(name), dt))
  }

  def resolveAtt(att: Attribute, schema: StructType): (ArgResolver, DataType) = att match {
    case ref: AttributeReference => columnRef(ref.name, ref.dataType) -> ref.dataType
    case att: UnresolvedAttribute =>
      schema.find(_.name == att.name) match {
        case None => throw new IllegalStateException(s"Can't resolve attr: $att")
        case Some(f) => columnRef(att.name, f.dataType) -> f.dataType
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
      case cns: CreateStruct => resolveStructChildren(cns.children)
      case cns: CreateNamedStruct => resolveStructChildren(cns.children.grouped(2).map(_.apply(1)).toSeq)
      case x => throw new IllegalArgumentException(s"Unexpected expression in column resolution: $x, ${x.getClass}")
    }
  }

}

trait UDFResolver {

  def applyUDF(
    column: String,
    udf: ScalaUDF,
    schema: StructType
  ): LocalTransform = {
    val resolvers = udf.children.map(c => ArgResolver.resolver(c, schema))

    val arity = resolvers.length
    (ds: PlainDataset) => {
      val accs = resolvers.map(r => r.mkAccessor(ds))
      val fn = arity match {
        case 1 => (i: Int) =>
          udf.function.asInstanceOf[Any => Any](accs(0).get(i))
        case 2 => (i: Int) =>
          udf.function.asInstanceOf[(Any, Any) => Any](accs(0).get(i), accs(1).get(i))
        case 3 => (i: Int) =>
          udf.function.asInstanceOf[(Any, Any, Any) => Any](accs(0).get(i), accs(1).get(i), accs(2).get(i))
        case 4 => (i: Int) =>
          udf.function.asInstanceOf[(Any, Any, Any, Any) => Any](accs(0).get(i), accs(1).get(i), accs(2).get(i), accs(3).get(i))
      }
      val values = (0 until ds.size).map(i => fn(i))
      ds.addColumn(Column(column, values))
    }
  }

}

