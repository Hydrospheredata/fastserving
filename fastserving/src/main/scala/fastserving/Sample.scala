package fastserving

import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

sealed trait Sample { self =>
  def mkDf(session: SparkSession): DataFrame
}

object Sample {

  case class EmptySample(schema: StructType) extends Sample {
    def mkDf(session: SparkSession): DataFrame =
      session.createDataFrame(session.sparkContext.emptyRDD[Row], schema)
  }

  case class RealSample(f: SparkSession => DataFrame) extends Sample {
    override def mkDf(session: SparkSession): DataFrame = f(session)
  }


  def empty(schema: StructType): EmptySample = EmptySample(schema)

  def real(f: SparkSession => DataFrame): RealSample = RealSample(f)
}
