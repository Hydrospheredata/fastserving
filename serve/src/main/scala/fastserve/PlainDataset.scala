package fastserve

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

}

