package fastserving.iterpr

import org.apache.spark.sql.types._

object Casts {

  def castFunction(from: DataType, to: DataType): Any => Any = to match {
    case DoubleType => castToDouble(from)
    case StringType => (a: Any) => a.toString
    case _ => throw new NotImplementedError(s"Cast to double not implemented for $from")
  }

  private def castToDouble(from: DataType): Any => Double = from match {
    case DoubleType => (x: Any) => x.asInstanceOf[Double]
    case IntegerType => (x: Any) => x.asInstanceOf[Int].toDouble
    case _ => throw new NotImplementedError(s"Cast to double not implemented for $from")
  }
}
