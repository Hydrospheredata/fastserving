package fastserving

import fastserving.iterpr.FastInterpreter
import org.apache.spark.ml.Transformer
import org.apache.spark.sql.SparkSession

sealed trait FastTransformer {
  def apply(d: PlainDataset): PlainDataset
  def compose(o: PlainDataset => PlainDataset): FastTransformer
  def andThen(o: PlainDataset => PlainDataset): FastTransformer
}

object ConstFastTransformer extends FastTransformer {
 def apply(d: PlainDataset): PlainDataset = d
 def compose(o: PlainDataset => PlainDataset): FastTransformer = FastTransformer(o)
 def andThen(o: PlainDataset => PlainDataset): FastTransformer = FastTransformer(o)
}

object FastTransformer {
  def apply(f: PlainDataset => PlainDataset): FastTransformer = new FastTransformer { self =>
    def apply(d: PlainDataset): PlainDataset = f(d)
    def compose(o: PlainDataset => PlainDataset): FastTransformer = FastTransformer((self.apply _ ).compose(o.apply))
    def andThen(o: PlainDataset => PlainDataset): FastTransformer = FastTransformer(f.andThen(o))
  }

  def build(t: Transformer, session: SparkSession, sample: Sample): FastTransformer = {
    val df = sample.mkDf(session)
    val schema = df.schema

    val out = t.transform(df)
    val plan = out.queryExecution.logical
    FastInterpreter.mkFastTransformer(plan, schema)
  }
}


