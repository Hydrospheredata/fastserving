package localserve.iterpr

import localserve.{Column, PlainDataset}
import org.apache.spark.sql.catalyst.expressions.Cast
import org.apache.spark.sql.types._

trait Casts {

  def applyCast(column: String, from: DataType, to: DataType): LocalTransform = {
    if (from == to) identity[PlainDataset] else mkCastFunc(column, from, to)
  }

  private def mkCastFunc(column: String, from: DataType, to: DataType):LocalTransform = to match {
    case DoubleType if from == DoubleType => identity[PlainDataset]
    case DoubleType =>
      (d: PlainDataset) => {
         val id = d.columnsId(column)
         val f = castToDouble(from)
         val values = d.columns(id).items.map(f)
         d.replace(Column(column, values))
      }
    case StringType if from == StringType => identity[PlainDataset]
    case _ => throw new NotImplementedError(s"Unsupported cast operation from: $from to: $to")
  }


  private def castToDouble(from: DataType): Any => Double = from match {
    case DoubleType => (x: Any) => x.asInstanceOf[Double]
    case IntegerType => (x: Any) => x.asInstanceOf[Int].toDouble
    case _ => throw new NotImplementedError(s"Cast to double not implemented for $from")
  }
}

object Casts extends Casts
