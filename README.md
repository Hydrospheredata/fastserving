# Fastserving

Spark-ml is built atop on DataFrames transformations.
Serving these models on small datasets is really slow and inefficient,
because it takes a lot of time to create and generate an execution plan for DataFrame for every request.

We already tried to wrap every model and separate its logic from dataframe usage in [spark-mk-serving](https://github.com/Hydrospheredata/spark-ml-serving)
and got a significant performance boost.
Here we used a more general approach based on alternative catalyst's ast interpretation.
To build a transformer we need to pass once an empty or real DataFrame sample into spark's transformer and then
interpret it's logical plan into FastTransformer.

Example:
```scala
import fastserve._

val session: SparkSession = _
// train or load model
val model: PipelineModel = _ 

val sample = Sample.empty(StructType(
  StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
))
val fastTransfomer = FastTransfromer.build(model, session, sample)

// serve
val ds = PlainDataSet(Column("features", Seq(Vector.dense(1.0)))
val out = fastTransfomer(ds)
```

Performance:

SringIndexer-VectorIndexer-RandomForestClassifier:
```
[info] Benchmark                               Mode  Cnt      Score      Error  Units
[info] StrIndVecIndRdnForestClassifier.fast   thrpt   20  60486.902 ± 1911.299  ops/s
[info] StrIndVecIndRdnForestClassifier.spark  thrpt   20     23.306 ±    1.028  ops/s
```
....
