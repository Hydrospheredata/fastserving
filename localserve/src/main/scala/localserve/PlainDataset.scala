package localserve

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

case class Column(name: String, items: Seq[Any])

class PlainDataset(val columns: Seq[Column]) {

  val columnsId: Map[String, Int] = columns.zipWithIndex.map({case (c, i) => c.name -> i}).toMap

  def size: Int = columns.headOption.map(_.items.length).getOrElse(0)

  def columnNames: Seq[String] = columnsId.keys.toSeq

  def columnByName(name: String): Column = columns(columnsId(name))

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
    case _ => false
  }

  override def toString: String = {

    def mkTable(): String = {
      def rowSeparator(colSizes: Seq[Int]): String = {
        colSizes.map("-" * _).mkString("+", "+", "+")
      }

      def rowFormat(items: Seq[String], colSizes: Seq[Int]): String = {
        items
          .zip(colSizes)
          .map((t) => if (t._2 == 0) "" else s"%${t._2}s".format(t._1))
          .mkString("|", "|", "|")
      }

      var stringParts = List.empty[String]

      val rowCount = (for (column <- columns) yield column.items.length).max
      val sizes = columns.map(
        column => (List(column.name) ++ column.items.map(_.toString)).map(_.length).max + 1
      )

      stringParts :+= rowSeparator(sizes)
      stringParts :+= rowFormat(columnNames, sizes)
      stringParts :+= rowSeparator(sizes)
      for (rowNumber <- List.range(0, rowCount)) {
        val row = columns.map { (column) =>
          if (column.items.lengthCompare(rowNumber) <= 0) {
            "–"
          } else {
            column.items(rowNumber).toString
          }
        }

        stringParts :+= rowFormat(row, sizes)
      }
      stringParts :+= rowSeparator(sizes)

      stringParts.mkString("\n")
    }
    if (size == 0) "PlainDataSet[empty]" else mkTable()
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
