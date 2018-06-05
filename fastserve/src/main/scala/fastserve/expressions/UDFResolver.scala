package fastserve.expressions

import fastserve.{Column, PlainDataset}
import org.apache.spark.sql.catalyst.analysis.UnresolvedAttribute
import org.apache.spark.sql.catalyst.expressions.{AttributeReference, Cast, ScalaUDF}
import org.apache.spark.sql.types.{DataType, StructType}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.catalyst.ScalaReflection

trait UDFResolver extends ColumnResolver {


  def applyUDF(column: String, udf: ScalaUDF, schema: StructType): LocalTransform = {
//    val inputPrep = udf.children.map(c => resolveColumn(c, schema))
//    val (prepT, names) = inputPrep.foldLeft((identity[PlainDataset] _, Seq.empty[String], schema)){
//      case ((t, n, s), (name, ps, Some(step))) =>  (t.andThen(step), (name +: names), schema.)
//      case ((t, n, s), (name, ps, None)) =>  t -> (name +: names)
//    }

    val (names, patchedS, prepT) = resolveColumns(udf.children, schema)
    println("WTF???")
    println(schema)
    println(udf.children)
    println(names)
    println(column)
    println()

    val maybeFields = names.map(n => n -> schema.find(_.name == n))
    val invalid = maybeFields.exists({case (n, f) => f.isEmpty})
    if (invalid) {
      val missing = maybeFields.collect({case (n, None) => n}).mkString(",")
      throw new IllegalStateException(s"Couldn't find schema for fiends: $missing")
    }
    val namesWithFields = maybeFields.collect({case (n, Some(f)) => n -> f.dataType}).toMap

    val udfF = convertUdf(udf.function, names.size)
    prepT.compose(d => {
      println(s"CALLL: $udf")
      println("In:")
      println(namesWithFields)
      println(d.toString)
      val accessors = namesWithFields.map({case (n, dt) => {
        val c = d.columns(d.columnsId(n))
        ItemAccessor(c, dt)
      }}).toSeq

      val values = (0 until d.size).map(i => {
        val in = accessors.map(_.get(i))
        val out = udfF(in)
        out
      })
      d.addColumn(Column(column, values))
    })
  }

  trait ItemAccessor {
    def get(i: Int): Any
  }

  object ItemAccessor {
    class DefaultAccessor(c: Column) extends ItemAccessor {
      override def get(i:Int): Any = c.items(i)
    }
    class CopyVectorAccessor(c: Column) extends ItemAccessor {
      override def get(i: Int): Any = c.items(i).asInstanceOf[Vector].copy
    }

    val VectorUDTType = ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType

    def apply(c: Column, d: DataType): ItemAccessor = {
      if (d == VectorUDTType) new CopyVectorAccessor(c) else new DefaultAccessor(c)
    }
  }

  private def convertUdf(f: Any, arity: Int): Seq[Any] => Any = arity match {
    case 1 => (seq: Seq[Any]) => f.asInstanceOf[Any => Any](seq(0))
    case 2 => (seq: Seq[Any]) => f.asInstanceOf[Any => Any](seq(0), seq(1))
    case 3 => (seq: Seq[Any]) => f.asInstanceOf[Any => Any](seq(0), seq(1), seq(2))
    case 4 => (seq: Seq[Any]) => f.asInstanceOf[Any => Any](seq(0), seq(1), seq(2), seq(4))
    case 5 => (seq: Seq[Any]) => f.asInstanceOf[Any => Any](seq(0), seq(1), seq(2), seq(4), seq(5))
    case x => throw new NotImplementedError(s"Udf convertation not implemented for $x arity")
  }
}

