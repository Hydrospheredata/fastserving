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
import fastserving._

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
[info] Benchmark                                          Mode  Cnt        Score       Error  Units
[info] Binarizer.fast                                    thrpt   20  1029521.883 ±  8764.727  ops/s
[info] Binarizer.spark                                   thrpt   20       45.108 ±     2.067  ops/s
[info] ChiSqSelector.fast                                thrpt   20   622923.447 ± 13885.723  ops/s
[info] ChiSqSelector.spark                               thrpt   20       42.384 ±     2.258  ops/s
[info] DCT.fast                                          thrpt   20   370858.941 ± 14844.250  ops/s
[info] DCT.spark                                         thrpt   20       44.665 ±     1.690  ops/s
[info] KMeans.fast                                       thrpt   20   300046.178 ±  4223.880  ops/s
[info] KMeans.spark                                      thrpt   20       43.562 ±     2.757  ops/s
[info] MaxAbsScaler.fast                                 thrpt   20   754863.381 ± 11686.241  ops/s
[info] MaxAbsScaler.spark                                thrpt   20       43.244 ±     1.475  ops/s
[info] MinMaxScaler.fast                                 thrpt   20   313526.232 ±  1958.355  ops/s
[info] MinMaxScaler.spark                                thrpt   20       44.220 ±     1.863  ops/s
[info] NGram.fast                                        thrpt   20   378002.699 ±  3907.919  ops/s
[info] NGram.spark                                       thrpt   20       29.004 ±     1.397  ops/s
[info] NaiveBayes.fast                                   thrpt   20   123969.446 ±   848.304  ops/s
[info] NaiveBayes.spark                                  thrpt   20       28.104 ±     1.328  ops/s
[info] Normalizer.fast                                   thrpt   20   887135.177 ± 47148.311  ops/s
[info] Normalizer.spark                                  thrpt   20       44.827 ±     4.062  ops/s
[info] PCA.spark                                         thrpt   20       42.103 ±     3.699  ops/s
[info] RegexTokenizer.fast                               thrpt   20   167620.933 ±  1024.652  ops/s
[info] RegexTokenizer.spark                              thrpt   20       31.327 ±     1.215  ops/s
[info] StandardScaler.fast                               thrpt   20   882237.545 ± 17899.279  ops/s
[info] StandardScaler.spark                              thrpt   20       44.238 ±     1.657  ops/s
[info] StopWordRemover.fast                              thrpt   20   712159.653 ±  4560.472  ops/s
[info] StopWordRemover.spark                             thrpt   20       27.423 ±     1.294  ops/s
[info] StrInd_VecInd_RandForestClassifier.fast           thrpt   20    73689.989 ±  2938.279  ops/s
[info] StrInd_VecInd_RandForestClassifier.spark          thrpt   20       22.088 ±     0.799  ops/s
[info] StringInd_VectorInd_DecisionTreeClassifier.fast   thrpt   20   101348.837 ±  4188.488  ops/s
[info] StringInd_VectorInd_DecisionTreeClassifier.spark  thrpt   20       22.679 ±     0.833  ops/s
[info] StringInd_VectorInd_GBTCLassifier.fast            thrpt   20   214604.858 ±  3108.330  ops/s
[info] StringInd_VectorInd_GBTCLassifier.spark           thrpt   20       28.959 ±     1.570  ops/s
[info] StringIndexer_OneHotEncoder.fast                  thrpt   20   381051.794 ±  9512.489  ops/s
[info] StringIndexer_OneHotEncoder.spark                 thrpt   20       35.368 ±     1.646  ops/s
[info] Tokenizer_HashingTF_IDF.fast                      thrpt   20    70017.999 ±  2777.558  ops/s
[info] Tokenizer_HashingTF_IDF.spark                     thrpt   20       22.353 ±     1.345  ops/s
[info] Tokenizer_HashingTF_LinearRegression.fast         thrpt   20    39377.643 ±   697.289  ops/s
[info] Tokenizer_HashingTF_LinearRegression.spark        thrpt   20       20.411 ±     1.251  ops/s
[info] VectorAssembler.fast                              thrpt   20   496876.825 ± 10774.071  ops/s
[info] VectorAssembler.spark                             thrpt   20       20.585 ±     0.840  ops/s
[info] VectorInd_DecisionTreeRegr.fast                   thrpt   20   307202.744 ± 14378.662  ops/s
[info] VectorInd_DecisionTreeRegr.spark                  thrpt   20       32.655 ±     1.347  ops/s
[info] VectorInd_GBTRegr.fast                            thrpt   20   229558.687 ±  2466.361  ops/s
[info] VectorInd_GBTRegr.spark                           thrpt   20       28.861 ±     2.091  ops/s
[info] VectorInd_RandomForestRegr.fast                   thrpt   20   136064.649 ±  2706.702  ops/s
[info] VectorInd_RandomForestRegr.spark                  thrpt   20       27.687 ±     1.764  ops/s
```
....
