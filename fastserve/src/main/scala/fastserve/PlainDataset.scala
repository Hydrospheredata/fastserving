package fastserve

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

case class Column(name: String, items: Seq[Any])

case class PlainDataset(
  columnsId: Map[String, Int],
  columns: Seq[Column]
) {

  def size: Int = columns.headOption.map(_.items.length).getOrElse(0)

  def addColumn(c: Column): PlainDataset = {
    copy(
      columnsId = columnsId + (c.name -> columns.size),
      columns = columns :+ c
    )
  }

  def replace(c: Column): PlainDataset = {
    val id = columnsId(c.name)
    copy(columns = columns.updated(id, c))
  }

  def toDataFrame(session: SparkSession, schema: StructType): DataFrame = {
    import session.implicits._

    val colls = columns.size
    val rows = (0 until size).map(y => {
      val rowData = (0 until columns.size).map(x => columns(y).items(x))
      Row(rowData: _*)
    })
    val rdd = session.sparkContext.parallelize(rows)
    session.createDataFrame(rdd, schema)
  }
}

object PlainDataset {

  val empty = PlainDataset(Map.empty, Seq.empty)

  def fromDataFrame(ds: DataFrame): PlainDataset = {
    val values = ds.collect()
    ds.columns.zipWithIndex.map({case (name, i) => Column(name, values.map(r => r.get(i))) })
      .foldLeft(empty)({case (d, c) => d.addColumn(c)})
  }

}
