package fastserve.expressions

import fastserve.{Column, PlainDataset}
import org.apache.spark.sql.catalyst.analysis.UnresolvedAttribute
import org.apache.spark.sql.catalyst.expressions.{AttributeReference, Cast, ScalaUDF}
import org.apache.spark.sql.types.StructType

trait UDFResolver extends ColumnResolver {

  def applyUDF(column: String, udf: ScalaUDF, schema: StructType): LocalTransform = {
    val inputPrep = udf.children.map(c => resolveColumn(c, schema))
    val (prepT, names) = inputPrep.foldLeft((identity[PlainDataset](_), Seq.empty[String])){
      case ((t, names), (name, Some(step))) =>  t.andThen(step) -> (name +: names)
      case ((t, names), (name, None)) =>  t -> (name +: names)
    }

    val udfF = convertUdf(udf.function, names.size)
    prepT.andThen(d => {
      val targets = names.map(n => d.columns(d.columnsId(n)))
      val values = (0 until d.size).map(i => {
        val in = targets.map(c => c.items(i))
        udfF(in)
      })
      d.addColumn(Column(column, values))
    })
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

