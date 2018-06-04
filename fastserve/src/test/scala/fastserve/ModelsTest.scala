package fastserve

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.scalatest.{FunSpec, Matchers}

class ModelsTest extends FunSpec with Matchers {

  val conf = new SparkConf()
    .setMaster("local[2]")
    .setAppName("test")
    .set("spark.ui.enabled", "false")

  val session: SparkSession = SparkSession.builder().config(conf).getOrCreate()

  modelTest(
    trainData = session.createDataFrame(Seq(
      (Vectors.dense(4.0, 0.2, 3.0, 4.0, 5.0), 1.0),
      (Vectors.dense(3.0, 0.3, 1.0, 4.1, 5.0), 1.0),
      (Vectors.dense(2.0, 0.5, 3.2, 4.0, 5.0), 1.0),
      (Vectors.dense(5.0, 0.7, 1.5, 4.0, 5.0), 1.0),
      (Vectors.dense(1.0, 0.1, 7.0, 4.0, 5.0), 0.0),
      (Vectors.dense(8.0, 0.3, 5.0, 1.0, 7.0), 0.0)
    )).toDF("features", "label"),
    stages = Seq(
      new StringIndexer().setInputCol("label").setOutputCol("indexedLabel"),
      new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4),
      new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)
    ),
    schema =
      StructType(
        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
      ),
    input = PlainDataset(
      columnsId = Map("features" -> 0),
      columns = Seq(
        Column("features", Seq(
          Vectors.dense(4.0, 0.2, 3.0, 4.0, 5.0),
          Vectors.dense(3.0, 0.3, 1.0, 4.1, 5.0),
          Vectors.dense(2.0, 0.5, 3.2, 4.0, 5.0),
          Vectors.dense(5.0, 0.7, 1.5, 4.0, 5.0),
          Vectors.dense(1.0, 0.1, 7.0, 4.0, 5.0),
          Vectors.dense(8.0, 0.3, 5.0, 1.0, 7.0)
        ))
      )
    )
  )


  modelTest(
    trainData = session.createDataFrame(
      Seq(
        (7, Vectors.dense(0.0, 0.0, 18.0, 1.0), 1.0),
        (8, Vectors.dense(0.0, 1.0, 12.0, 0.0), 0.0),
        (9, Vectors.dense(1.0, 0.0, 15.0, 0.1), 0.0)
      )
    ).toDF("id", "features", "clicked"),
    stages = Seq(
      new ChiSqSelector().setNumTopFeatures(1).setFeaturesCol("features").setLabelCol("clicked").setOutputCol("selectedFeatures")
    ),
    schema =
      StructType(
        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
      ),
    input = PlainDataset(
      columnsId = Map("features" -> 0),
      columns = Seq(
        Column("features", Seq(
          Vectors.dense(0.0, 0.0, 18.0, 1.0),
          Vectors.dense(0.0, 1.0, 12.0, 0.0),
          Vectors.dense(1.0, 0.0, 15.0, 0.1)
        ))
      )
    )
  )
//
//  modelTest(
//    trainData = session.createDataFrame(
//      Seq(
//        (0.0, "Hi I heard about Spark"),
//        (0.0, "I wish Java could use case classes"),
//        (1.0, "Logistic regression models are neat")
//      )
//    ).toDF("label", "sentence"),
//    stages = Seq(
//      new Tokenizer().setInputCol("sentence").setOutputCol("words"),
//      new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20),
//      new IDF().setInputCol("rawFeatures").setOutputCol("features")
//    ),
//    schema =
//      StructType(
//        StructField("sentence", StringType) :: Nil
//      ),
//    input = PlainDataset(
//      columnsId = Map("sentence" -> 0),
//      columns = Seq(
//        Column("sentence", Seq(
//          "Hi I heard about Spark",
//          "I wish Java could use case classes",
//          "Logistic regression models are neat"
//        ))
//      )
//    )
//  )
//
//  modelTest(
//    trainData = session.createDataFrame(Seq(
//      (0, Array("Provectus", "is", "such", "a", "cool", "company")),
//      (1, Array("Big", "data", "rules", "the", "world")),
//      (2, Array("Cloud", "solutions", "are", "our", "future"))
//    )).toDF("id", "words"),
//    stages = Seq(
//      new NGram().setN(2).setInputCol("words").setOutputCol("ngrams")
//    ),
//    schema =
//      StructType(
//        StructField("words", new ArrayType(StringType, false)) :: Nil
//      ),
//    input = PlainDataset(
//      columnsId = Map("words" -> 0),
//      columns = Seq(
//        Column("words", Seq(
//          Array("Provectus", "is", "such", "a", "cool", "company"),
//          Array("Big", "data", "rules", "the", "world"),
//          Array("Cloud", "solutions", "are", "our", "future")
//        ))
//      )
//    )
//  )
//
//  modelTest(
//    trainData = session.createDataFrame(Seq(
//      Vectors.dense(0.0, 10.3, 1.0, 4.0, 5.0),
//      Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
//      Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
//    ).map(Tuple1.apply)).toDF("features"),
//    stages = Seq(
//      new StandardScaler()
//        .setInputCol("features")
//        .setOutputCol("scaledFeatures")
//        .setWithStd(true)
//        .setWithMean(false)
//    ),
//    schema =
//      StructType(
//        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
//      ),
//    input = PlainDataset(
//      columnsId = Map("features" -> 0),
//      columns = Seq(
//        Column("features", Seq(
//          Vectors.dense(0.0, 10.3, 1.0, 4.0, 5.0),
//          Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
//          Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
//        ))
//      )
//    )
//  )
//
//  modelTest(
//    trainData = session.createDataFrame(Seq(
//      (0, Seq("I", "saw", "the", "red", "balloon")),
//      (1, Seq("Mary", "had", "a", "little", "lamb"))
//    )).toDF("id", "raw"),
//    stages = Seq(
//      new StopWordsRemover()
//        .setInputCol("raw")
//        .setOutputCol("filtered")
//    ),
//    schema =
//      StructType(
//        StructField("raw", new ArrayType(StringType, false)) :: Nil
//      ),
//    input = PlainDataset(
//      columnsId = Map("raw" -> 0),
//      columns = Seq(
//        Column("raw", Seq(
//          Seq("I", "saw", "the", "red", "balloon"),
//          Seq("Mary", "had", "a", "little", "lamb")
//        ))
//      )
//    )
//  )
//
//  modelTest(
//    trainData = session.createDataFrame(Seq(
//      (0, Vectors.dense(1.0, 0.1, -8.0)),
//      (1, Vectors.dense(2.0, 1.0, -4.0)),
//      (2, Vectors.dense(4.0, 10.0, 8.0))
//    )).toDF("id", "features"),
//    stages = Seq(
//      new MaxAbsScaler()
//        .setInputCol("features")
//        .setOutputCol("scaledFeatures")
//    ),
//    schema =
//      StructType(
//        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
//      ),
//    input = PlainDataset(
//      columnsId = Map("features" -> 0),
//      columns = Seq(
//        Column("features", Seq(
//          Vectors.dense(1.0, 0.1, -8.0),
//          Vectors.dense(2.0, 1.0, -4.0),
//          Vectors.dense(4.0, 10.0, 8.0)
//        ))
//      )
//    )
//  )
//
//  modelTest(
//    trainData = session.createDataFrame(Seq(
//      (0, Vectors.dense(1.0, 0.1, -1.0)),
//      (1, Vectors.dense(2.0, 1.1, 1.0)),
//      (2, Vectors.dense(3.0, 10.1, 3.0))
//    )).toDF("id", "features"),
//    stages = Seq(
//      new MinMaxScaler()
//        .setInputCol("features")
//        .setOutputCol("scaledFeatures")
//    ),
//    schema =
//      StructType(
//        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
//      ),
//    input = PlainDataset(
//      columnsId = Map("features" -> 0),
//      columns = Seq(
//        Column("features", Seq(
//          Vectors.dense(1.0, 0.1, -1.0),
//          Vectors.dense(2.0, 1.1, 1.0),
//          Vectors.dense(3.0, 10.1, 3.0)
//        ))
//      )
//    )
//  )
//
//  modelTest(
//    trainData = session.createDataFrame(Seq(
//      (0, "a"), (1, "b"), (2, "c"),
//      (3, "a"), (4, "a"), (5, "c")
//    )).toDF("id", "category"),
//    stages = Seq(
//      new StringIndexer()
//        .setInputCol("category")
//        .setOutputCol("categoryIndex"),
//      new OneHotEncoder()
//        .setInputCol("categoryIndex")
//        .setOutputCol("categoryVec")
//    ),
//    schema =
//      StructType(
//        StructField("category", StringType) :: Nil
//      ),
//    input = PlainDataset(
//      columnsId = Map("category" -> 0),
//      columns = Seq(
//        Column("category", Seq(
//          "a", "b", "c", "a", "a", "c"
//        ))
//      )
//    )
//  )
//
//  modelTest(
//    trainData = session.createDataFrame(Seq(
//      Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
//      Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
//      Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
//    ).map(Tuple1.apply)).toDF("features"),
//    stages = Seq(
//      new PCA()
//        .setInputCol("features")
//        .setOutputCol("pcaFeatures")
//        .setK(3)
//    ),
//    schema =
//      StructType(
//        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
//      ),
//    input = PlainDataset(
//      columnsId = Map("features" -> 0),
//      columns = Seq(
//        Column("features", Seq(
//          Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
//          Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
//          Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
//        ))
//      )
//    )
//  )
//
//  modelTest(
//    trainData = session.createDataFrame(Seq(
//      (0, Vectors.dense(1.0, 0.5, -1.0)),
//      (1, Vectors.dense(2.0, 1.0, 1.0)),
//      (2, Vectors.dense(4.0, 10.0, 2.0))
//    )).toDF("id", "features"),
//    stages = Seq(
//      new Normalizer()
//        .setInputCol("features")
//        .setOutputCol("normFeatures")
//        .setP(1.0)
//    ),
//    schema =
//      StructType(
//        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
//      ),
//    input = PlainDataset(
//      columnsId = Map("features" -> 0),
//      columns = Seq(
//        Column("features", Seq(
//          Vectors.dense(1.0, 0.5, -1.0),
//          Vectors.dense(2.0, 1.0, 1.0),
//          Vectors.dense(4.0, 10.0, 2.0)
//        ))
//      )
//    )
//  )

//  modelTest(
//    data = session.createDataFrame(Seq(
//      Vectors.dense(0.0, 1.0, -2.0, 3.0),
//      Vectors.dense(-1.0, 2.0, 4.0, -7.0),
//      Vectors.dense(14.0, -2.0, -5.0, 1.0)
//    ).map(Tuple1.apply)).toDF("features"),
//    steps = Seq(
//      new DCT()
//        .setInputCol("features")
//        .setOutputCol("featuresDCT")
//        .setInverse(false)
//    ),
//    columns = Seq(
//      "featuresDCT"
//    )
//  )
//
//  modelTest(
//    data = session.createDataFrame(Seq(
//      (Vectors.dense(4.0, 0.2, 3.0, 4.0, 5.0), 1.0),
//      (Vectors.dense(3.0, 0.3, 1.0, 4.1, 5.0), 1.0),
//      (Vectors.dense(2.0, 0.5, 3.2, 4.0, 5.0), 1.0),
//      (Vectors.dense(5.0, 0.7, 1.5, 4.0, 5.0), 1.0),
//      (Vectors.dense(1.0, 0.1, 7.0, 4.0, 5.0), 0.0),
//      (Vectors.dense(8.0, 0.3, 5.0, 1.0, 7.0), 0.0)
//    )).toDF("features", "label"),
//    steps = Seq(
//      new NaiveBayes()
//    ),
//    columns = Seq(
//      "prediction"
//    )
//  )
//
//  modelTest(
//    data = session.createDataFrame(Seq(
//      (0, 0.1),
//      (1, 0.8),
//      (2, 0.2)
//    )).toDF("id", "feature"),
//    steps = Seq(
//      new Binarizer()
//        .setInputCol("feature")
//        .setOutputCol("binarized_feature")
//        .setThreshold(5.0)
//    ),
//    columns = Seq(
//      "binarized_feature"
//    )
//  )
//
//  modelTest(
//    data = session.createDataFrame(Seq(
//      (Vectors.dense(4.0, 0.2, 3.0, 4.0, 5.0), 1.0),
//      (Vectors.dense(3.0, 0.3, 1.0, 4.1, 5.0), 1.0),
//      (Vectors.dense(2.0, 0.5, 3.2, 4.0, 5.0), 1.0),
//      (Vectors.dense(5.0, 0.7, 1.5, 4.0, 5.0), 1.0),
//      (Vectors.dense(1.0, 0.1, 7.0, 4.0, 5.0), 0.0),
//      (Vectors.dense(8.0, 0.3, 5.0, 1.0, 7.0), 0.0)
//    )).toDF("features", "label"),
//    steps =
//      Seq(
//        new StringIndexer()
//          .setInputCol("label")
//          .setOutputCol("indexedLabel"),
//        new VectorIndexer()
//          .setInputCol("features")
//          .setOutputCol("indexedFeatures")
//          .setMaxCategories(4),
//        new GBTClassifier()
//          .setLabelCol("indexedLabel")
//          .setFeaturesCol("indexedFeatures")
//          .setMaxIter(10)
//      ),
//    columns = Seq(
//      "prediction"
//    )
//  )
//
//  modelTest(
//    data = session.createDataFrame(Seq(
//      (Vectors.dense(4.0, 0.2, 3.0, 4.0, 5.0), 1.0),
//      (Vectors.dense(3.0, 0.3, 1.0, 4.1, 5.0), 1.0),
//      (Vectors.dense(2.0, 0.5, 3.2, 4.0, 5.0), 1.0),
//      (Vectors.dense(5.0, 0.7, 1.5, 4.0, 5.0), 1.0),
//      (Vectors.dense(1.0, 0.1, 7.0, 4.0, 5.0), 0.0),
//      (Vectors.dense(8.0, 0.3, 5.0, 1.0, 7.0), 0.0)
//    )).toDF("features", "label"),
//    steps = Seq(
//      new StringIndexer()
//        .setInputCol("label")
//        .setOutputCol("indexedLabel"),
//      new VectorIndexer()
//        .setInputCol("features")
//        .setOutputCol("indexedFeatures")
//        .setMaxCategories(4),
//      new DecisionTreeClassifier()
//        .setLabelCol("indexedLabel")
//        .setFeaturesCol("indexedFeatures")
//    ),
//    columns = Seq(
//      "prediction"
//    )
//  )
//
//  modelTest(
//    data = session.createDataFrame(Seq(
//      (0L, "a b c d e spark", 1.0),
//      (1L, "b d", 0.0),
//      (2L, "spark f g h", 1.0),
//      (3L, "hadoop mapreduce", 0.0)
//    )).toDF("id", "text", "label"),
//    steps = Seq(
//      new Tokenizer().setInputCol("text").setOutputCol("words"),
//      new HashingTF().setNumFeatures(1000).setInputCol("words").setOutputCol("features"),
//      new LinearRegression()
//        .setMaxIter(10)
//        .setRegParam(0.3)
//        .setElasticNetParam(0.8)
//    ),
//    columns = Seq(
//      "prediction"
//    )
//  )
//
//  modelTest(
//    data = session.createDataFrame(Seq(
//      (Vectors.dense(4.0, 0.2, 3.0, 4.0, 5.0), 1.0),
//      (Vectors.dense(3.0, 0.3, 1.0, 4.1, 5.0), 1.0),
//      (Vectors.dense(2.0, 0.5, 3.2, 4.0, 5.0), 1.0),
//      (Vectors.dense(5.0, 0.7, 1.5, 4.0, 5.0), 1.0),
//      (Vectors.dense(1.0, 0.1, 7.0, 4.0, 5.0), 0.0),
//      (Vectors.dense(8.0, 0.3, 5.0, 1.0, 7.0), 0.0)
//    )).toDF("features", "label"),
//    steps = Seq(
//      new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4),
//      new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures")
//    ),
//    columns = Seq(
//      "prediction"
//    )
//  )
//
//  modelTest(
//    data = session.createDataFrame(Seq(
//      (Vectors.dense(4.0, 0.2, 3.0, 4.0, 5.0), 1.0),
//      (Vectors.dense(3.0, 0.3, 1.0, 4.1, 5.0), 1.0),
//      (Vectors.dense(2.0, 0.5, 3.2, 4.0, 5.0), 1.0),
//      (Vectors.dense(5.0, 0.7, 1.5, 4.0, 5.0), 1.0),
//      (Vectors.dense(1.0, 0.1, 7.0, 4.0, 5.0), 0.0),
//      (Vectors.dense(8.0, 0.3, 5.0, 1.0, 7.0), 0.0)
//    )).toDF("features", "label"),
//    steps = Seq(
//      new StringIndexer().setInputCol("label").setOutputCol("indexedLabel"),
//      new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4),
//      new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)
//    ),
//    columns = Seq(
//      "prediction"
//    )
//  )
//
//  modelTest(
//    data = session.createDataFrame(Seq(
//      (Vectors.dense(4.0, 0.2, 3.0, 4.0, 5.0), 1.0),
//      (Vectors.dense(3.0, 0.3, 1.0, 4.1, 5.0), 1.0),
//      (Vectors.dense(2.0, 0.5, 3.2, 4.0, 5.0), 1.0),
//      (Vectors.dense(5.0, 0.7, 1.5, 4.0, 5.0), 1.0),
//      (Vectors.dense(1.0, 0.1, 7.0, 4.0, 5.0), 0.0),
//      (Vectors.dense(8.0, 0.3, 5.0, 1.0, 7.0), 0.0)
//    )).toDF("features", "label"),
//    steps = Seq(
//      new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4),
//      new RandomForestRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures")
//    ),
//    columns = Seq(
//      "prediction"
//    )
//  )
//
//
//  modelTest(
//    data = session.createDataFrame(Seq(
//      (Vectors.dense(4.0, 0.2, 3.0, 4.0, 5.0), 1.0),
//      (Vectors.dense(3.0, 0.3, 1.0, 4.1, 5.0), 1.0),
//      (Vectors.dense(2.0, 0.5, 3.2, 4.0, 5.0), 1.0),
//      (Vectors.dense(5.0, 0.7, 1.5, 4.0, 5.0), 1.0),
//      (Vectors.dense(1.0, 0.1, 7.0, 4.0, 5.0), 0.0),
//      (Vectors.dense(8.0, 0.3, 5.0, 1.0, 7.0), 0.0)
//    )).toDF("features", "label"),
//    steps = Seq(
//      new VectorIndexer()
//        .setInputCol("features")
//        .setOutputCol("indexedFeatures")
//        .setMaxCategories(4),
//      new GBTRegressor()
//        .setLabelCol("label")
//        .setFeaturesCol("indexedFeatures")
//        .setMaxIter(10)
//    ),
//    columns = Seq(
//      "prediction"
//    )
//  )
//
//  modelTest(
//    data = session.createDataFrame(Seq(
//      (Vectors.dense(4.0, 0.2, 3.0, 4.0, 5.0), 1.0),
//      (Vectors.dense(3.0, 0.3, 1.0, 4.1, 5.0), 1.0),
//      (Vectors.dense(2.0, 0.5, 3.2, 4.0, 5.0), 1.0),
//      (Vectors.dense(5.0, 0.7, 1.5, 4.0, 5.0), 1.0),
//      (Vectors.dense(1.0, 0.1, 7.0, 4.0, 5.0), 0.0),
//      (Vectors.dense(8.0, 0.3, 5.0, 1.0, 7.0), 0.0)
//    )).toDF("features", "label"),
//    steps = Seq(
//      new KMeans().setK(2).setSeed(1L)
//    ),
//    columns = Seq(
//      "prediction"
//    )
//  )
//
//  modelTest(
//    data = session.createDataFrame(Seq(
//      (Vectors.dense(4.0, 0.2, 3.0, 4.0, 5.0), 1.0),
//      (Vectors.dense(3.0, 0.3, 1.0, 4.1, 5.0), 1.0),
//      (Vectors.dense(2.0, 0.5, 3.2, 4.0, 5.0), 1.0),
//      (Vectors.dense(5.0, 0.7, 1.5, 4.0, 5.0), 1.0),
//      (Vectors.dense(1.0, 0.1, 7.0, 4.0, 5.0), 0.0),
//      (Vectors.dense(8.0, 0.3, 5.0, 1.0, 7.0), 0.0)
//    )).toDF("features", "label"),
//    steps = Seq(
//      new GaussianMixture().setK(2)
//    ),
//    columns = Seq(
//      "prediction"
//    )
//  )
//
//  modelTest(
//    data = session.createDataFrame(Seq(
//      (0, "Hi I heard about Spark"),
//      (1, "I wish Java could use case classes"),
//      (2, "Logistic,regression,models,are,neat")
//    )).toDF("id", "sentence"),
//    steps = Seq(
//      new RegexTokenizer()
//        .setInputCol("sentence")
//        .setOutputCol("words")
//        .setPattern("\\W")
//    ),
//    columns = Seq(
//      "words"
//    )
//  )
//
//  modelTest(
//    data = session.createDataFrame(Seq((0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0))
//    ).toDF("id", "hour", "mobile", "userFeatures", "clicked"),
//    steps = Seq(
//      new VectorAssembler()
//        .setInputCols(Array("hour", "mobile", "userFeatures"))
//        .setOutputCol("features")
//    ),
//    columns = Seq(
//      "features"
//    )
//  )
//
//  modelTest(
//    data = session.read.format("libsvm")
//      .load(getClass.getResource("/data/mllib/sample_lda_libsvm_data.txt").getPath),
//    steps = Seq(
//      new LDA().setK(10).setMaxIter(10)
//    ),
//    columns = Seq(
//      "topicDistribution"
//    ),
//    accuracy = 1
//  )


  
  def modelTest(
    trainData: DataFrame,
    stages: Seq[PipelineStage],
    schema: StructType,
    input: PlainDataset
  ):Unit = {
    val name = stages.map(_.getClass.getSimpleName).foldLeft("") {
      case ("", b) => b
      case (a, b) => a + "-" + b
    }

    def compare(pd: PlainDataset, df: DataFrame): Unit = {
    }

    it(name) {
      val pipeline = new Pipeline().setStages(stages.toArray)
      val pipelineModel = pipeline.fit(trainData)

      val emptyDf = session.createDataFrame(session.sparkContext.emptyRDD[Row], schema)
      val transformer = FastInterpreter.fromTransformer(pipelineModel, emptyDf)

      val out = transformer(input)

      val origDf = pipelineModel.transform(input.toDataFrame(session, schema))
      val origRows = origDf.collect()

      out.columnsId.keys.toSeq should contain theSameElementsAs origDf.columns.toSeq
      out.size shouldBe origRows.length

    }
  }
}
