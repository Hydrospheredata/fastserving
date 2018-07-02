package fastserving.interp

import fastserving.interp.compat.UDFChildrenCompat
import fastserving.{Column, PlainDataset}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.catalyst.expressions.{GenericRow, ScalaUDF}
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

}

trait UDFResolver {

  def applyUDF(
    column: String,
    udf: ScalaUDF,
    schema: StructType
  ): LocalTransform = {
    val resolvers = udf.children.map(c => UDFChildrenCompat.resolver(c, schema))

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

