package fastserve

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

case class Column(name: String, items: Seq[Any])

class PlainDataset(val columns: Seq[Column]) {

  val columnsId = columns.zipWithIndex.map({case (c, i) => c.name -> i})

  def size: Int = columns.headOption.map(_.items.length).getOrElse(0)

  def addColumn(c: Column): PlainDataset = {
    new PlainDataset(columns :+ c)
  }

  def replace(c: Column): PlainDataset = {
    new PlainDataset(columns.map(o => if (o.name == c.name) c else o))
  }

  def toDataFrame(session: SparkSession, schema: StructType): DataFrame = {
    import session.implicits._

    val colls = columns.size
    val rows = (0 until size).map(y => {
      val rowData = (0 until columns.size).map(x => columns(x).items(y))
      Row(rowData: _*)
    })
    val rdd = session.sparkContext.parallelize(rows)
    session.createDataFrame(rdd, schema)
  }

  override def hashCode(): Int = columns.hashCode()

  override def equals(obj: Any): Boolean = obj match {
    case PlainDataset(ocol) => ocol.sortBy(_.name).equals(columns.sortBy(_.name))
    case oth => false
  }
}

object PlainDataset {

  val empty = PlainDataset(Seq.empty: _*)

  def apply(columns: Column*): PlainDataset =
    new PlainDataset(columns)

  def unapply(arg: PlainDataset): Option[Seq[Column]] = Some(arg.columns)

  def fromDataFrame(ds: DataFrame): PlainDataset = {
    val values = ds.collect()
    ds.columns.zipWithIndex.map({case (name, i) => Column(name, values.map(r => r.get(i))) })
      .foldLeft(empty)({case (d, c) => d.addColumn(c)})
  }

}
