package fastserve.expressions

import fastserve.{Column, PlainDataset}
import org.apache.spark.sql.catalyst.expressions.Cast
import org.apache.spark.sql.types._

trait Casts {

  def applyCast(column: String, from: DataType, to: DataType): LocalTransform = to match {
    case DoubleType =>
      (d: PlainDataset) => {
         val id = d.columnsId(column)
         val f = castToDouble(from)
         val values = d.columns(id).items.map(f)
         d.replace(Column(column, values))
      }
    case x => throw new NotImplementedError(s"Unsupported cast operation: $x")
  }


  private def castToDouble(from: DataType): Any => Double = from match {
    case DoubleType => (x: Any) => x.asInstanceOf[Double]
    case IntegerType => (x: Any) => x.asInstanceOf[Int].toDouble
    case x => throw new NotImplementedError(s"Cast to double not implemented for $from")
  }
}

object Casts extends Casts
