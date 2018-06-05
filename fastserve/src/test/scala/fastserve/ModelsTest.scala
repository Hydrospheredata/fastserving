package fastserve

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.{DecisionTreeClassifier, GBTClassifier, NaiveBayes, RandomForestClassifier}
import org.apache.spark.ml.clustering.{GaussianMixture, KMeans, LDA}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{DenseVector, VectorUDT, Vectors}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.regression.{DecisionTreeRegressor, GBTRegressor, LinearRegression, RandomForestRegressor}
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.types._
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
      Column("features", Seq(
        Vectors.dense(0.0, 0.0, 18.0, 1.0),
        Vectors.dense(0.0, 1.0, 12.0, 0.0),
        Vectors.dense(1.0, 0.0, 15.0, 0.1)
      ))
    )
  )

  modelTest(
    trainData = session.createDataFrame(
      Seq(
        (0.0, "Hi I heard about Spark"),
        (0.0, "I wish Java could use case classes"),
        (1.0, "Logistic regression models are neat")
      )
    ).toDF("label", "sentence"),
    stages = Seq(
      new Tokenizer().setInputCol("sentence").setOutputCol("words"),
      new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20),
      new IDF().setInputCol("rawFeatures").setOutputCol("features")
    ),
    schema =
      StructType(
        StructField("sentence", StringType) :: Nil
      ),
    input = PlainDataset(
      Column("sentence", Seq(
        "Hi I heard about Spark",
        "I wish Java could use case classes",
        "Logistic regression models are neat"
      ))
    )
  )

  modelTest(
    trainData = session.createDataFrame(Seq(
      (0, Array("Hydrosphere", "is", "such", "a", "cool", "company")),
      (1, Array("Big", "data", "rules", "the", "world")),
      (2, Array("Cloud", "solutions", "are", "our", "future"))
    )).toDF("id", "words"),
    stages = Seq(
      new NGram().setN(2).setInputCol("words").setOutputCol("ngrams")
    ),
    schema =
      StructType(
        StructField("words", new ArrayType(StringType, false)) :: Nil
      ),
    input = PlainDataset(
      Column("words", Seq(
        Seq("Provectus", "is", "such", "a", "cool", "company"),
        Seq("Big", "data", "rules", "the", "world"),
        Seq("Cloud", "solutions", "are", "our", "future")
      ))
    )
  )

  modelTest(
    trainData = session.createDataFrame(Seq(
      Vectors.dense(0.0, 10.3, 1.0, 4.0, 5.0),
      Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
      Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
    ).map(Tuple1.apply)).toDF("features"),
    stages = Seq(
      new StandardScaler()
        .setInputCol("features")
        .setOutputCol("scaledFeatures")
        .setWithStd(true)
        .setWithMean(false)
    ),
    schema =
      StructType(
        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
      ),
    input = PlainDataset(
      Column("features", Seq(
        Vectors.dense(0.0, 10.3, 1.0, 4.0, 5.0),
        Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
        Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
      ))
    )
  )

  modelTest(
    trainData = session.createDataFrame(Seq(
      (0, Seq("I", "saw", "the", "red", "balloon")),
      (1, Seq("Mary", "had", "a", "little", "lamb"))
    )).toDF("id", "raw"),
    stages = Seq(
      new StopWordsRemover()
        .setInputCol("raw")
        .setOutputCol("filtered")
    ),
    schema =
      StructType(
        StructField("raw", new ArrayType(StringType, false)) :: Nil
      ),
    input = PlainDataset(
      Column("raw", Seq(
        Seq("I", "saw", "the", "red", "balloon"),
        Seq("Mary", "had", "a", "little", "lamb")
      ))
    )
  )

  modelTest(
    trainData = session.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 0.1, -8.0)),
      (1, Vectors.dense(2.0, 1.0, -4.0)),
      (2, Vectors.dense(4.0, 10.0, 8.0))
    )).toDF("id", "features"),
    stages = Seq(
      new MaxAbsScaler()
        .setInputCol("features")
        .setOutputCol("scaledFeatures")
    ),
    schema =
      StructType(
        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
      ),
    input = PlainDataset(
      Column("features", Seq(
        Vectors.dense(1.0, 0.1, -8.0),
        Vectors.dense(2.0, 1.0, -4.0),
        Vectors.dense(4.0, 10.0, 8.0)
      ))
    )
  )

  modelTest(
    trainData = session.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 0.1, -1.0)),
      (1, Vectors.dense(2.0, 1.1, 1.0)),
      (2, Vectors.dense(3.0, 10.1, 3.0))
    )).toDF("id", "features"),
    stages = Seq(
      new MinMaxScaler()
        .setInputCol("features")
        .setOutputCol("scaledFeatures")
    ),
    schema =
      StructType(
        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
      ),
    input = PlainDataset(
      Column("features", Seq(
        Vectors.dense(1.0, 0.1, -1.0),
        Vectors.dense(2.0, 1.1, 1.0),
        Vectors.dense(3.0, 10.1, 3.0)
      ))
    ))

  modelTest(
    trainData = session.createDataFrame(Seq(
      (0, "a"), (1, "b"), (2, "c"),
      (3, "a"), (4, "a"), (5, "c")
    )).toDF("id", "category"),
    stages = Seq(
      new StringIndexer()
        .setInputCol("category")
        .setOutputCol("categoryIndex"),
      new OneHotEncoder()
        .setInputCol("categoryIndex")
        .setOutputCol("categoryVec")
    ),
    schema =
      StructType(
        StructField("category", StringType) :: Nil
      ),
    input = PlainDataset(
      Column("category", Seq(
        "a", "b", "c", "a", "a", "c"
      ))
    )
  )

  modelTest(
    trainData = session.createDataFrame(Seq(
      Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
      Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
      Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
    ).map(Tuple1.apply)).toDF("features"),
    stages = Seq(
      new PCA()
        .setInputCol("features")
        .setOutputCol("pcaFeatures")
        .setK(3)
    ),
    schema =
      StructType(
        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
      ),
    input = PlainDataset(
      Column("features", Seq(
        Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
        Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
        Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
      ))
    )
  )

  modelTest(
    trainData = session.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 0.5, -1.0)),
      (1, Vectors.dense(2.0, 1.0, 1.0)),
      (2, Vectors.dense(4.0, 10.0, 2.0))
    )).toDF("id", "features"),
    stages = Seq(
      new Normalizer()
        .setInputCol("features")
        .setOutputCol("normFeatures")
        .setP(1.0)
    ),
    schema =
      StructType(
        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
      ),
    input = PlainDataset(
      Column("features", Seq(
        Vectors.dense(1.0, 0.5, -1.0),
        Vectors.dense(2.0, 1.0, 1.0),
        Vectors.dense(4.0, 10.0, 2.0)
      ))
    )
  )

  modelTest(
    trainData = session.createDataFrame(Seq(
      Vectors.dense(0.0, 1.0, -2.0, 3.0),
      Vectors.dense(-1.0, 2.0, 4.0, -7.0),
      Vectors.dense(14.0, -2.0, -5.0, 1.0)
    ).map(Tuple1.apply)).toDF("features"),
    stages = Seq(
      new DCT()
        .setInputCol("features")
        .setOutputCol("featuresDCT")
        .setInverse(false)
    ),
    schema =
      StructType(
        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
      ),
    input = PlainDataset(
      Column("features", Seq(
        Vectors.dense(0.0, 1.0, -2.0, 3.0),
        Vectors.dense(-1.0, 2.0, 4.0, -7.0),
        Vectors.dense(14.0, -2.0, -5.0, 1.0)
      ))
    )
  )

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
      new NaiveBayes()
    ),
    schema =
      StructType(
        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
      ),
    input = PlainDataset(
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

  modelTest(
    trainData = session.createDataFrame(Seq(
      (0, 0.1),
      (1, 0.8),
      (2, 0.2)
    )).toDF("id", "feature"),
    stages = Seq(
      new Binarizer()
        .setInputCol("feature")
        .setOutputCol("binarized_feature")
        .setThreshold(5.0)
    ),
    schema =
      StructType(
        StructField("feature", DoubleType) :: Nil
      ),
    input = PlainDataset(
      Column("feature", Seq(0.1, 0.8, 0.2))
    )
  )

  modelTest(
    trainData = session.createDataFrame(Seq(
      (Vectors.dense(4.0, 0.2, 3.0, 4.0, 5.0), 1.0),
      (Vectors.dense(3.0, 0.3, 1.0, 4.1, 5.0), 1.0),
      (Vectors.dense(2.0, 0.5, 3.2, 4.0, 5.0), 1.0),
      (Vectors.dense(5.0, 0.7, 1.5, 4.0, 5.0), 1.0),
      (Vectors.dense(1.0, 0.1, 7.0, 4.0, 5.0), 0.0),
      (Vectors.dense(8.0, 0.3, 5.0, 1.0, 7.0), 0.0)
    )).toDF("features", "label"),
    stages =
      Seq(
        new StringIndexer()
          .setInputCol("label")
          .setOutputCol("indexedLabel"),
        new VectorIndexer()
          .setInputCol("features")
          .setOutputCol("indexedFeatures")
          .setMaxCategories(4),
        new GBTClassifier()
          .setLabelCol("indexedLabel")
          .setFeaturesCol("indexedFeatures")
          .setMaxIter(10)
      ),
    schema =
      StructType(
        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
      ),
    input = PlainDataset(
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
      new StringIndexer()
        .setInputCol("label")
        .setOutputCol("indexedLabel"),
      new VectorIndexer()
        .setInputCol("features")
        .setOutputCol("indexedFeatures")
        .setMaxCategories(4),
      new DecisionTreeClassifier()
        .setLabelCol("indexedLabel")
        .setFeaturesCol("indexedFeatures")
    ),
    schema =
      StructType(
        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
      ),
    input = PlainDataset(
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

  modelTest(
    trainData = session.createDataFrame(Seq(
      (0L, "a b c d e spark", 1.0),
      (1L, "b d", 0.0),
      (2L, "spark f g h", 1.0),
      (3L, "hadoop mapreduce", 0.0)
    )).toDF("id", "text", "label"),
    stages = Seq(
      new Tokenizer().setInputCol("text").setOutputCol("words"),
      new HashingTF().setNumFeatures(1000).setInputCol("words").setOutputCol("features"),
      new LinearRegression()
        .setMaxIter(10)
        .setRegParam(0.3)
        .setElasticNetParam(0.8)
    ),
    schema =
      StructType(
        StructField("text", StringType) :: Nil
      ),
    input = PlainDataset(
      Column("text", Seq(
        "a b c d e spark",
        "b d",
        "spark f g h",
        "hadoop mapreduce"
      ))
    )
  )

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
      new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4),
      new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures")
    ),
    schema =
      StructType(
        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
      ),
    input = PlainDataset(
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
      new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4),
      new RandomForestRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures")
    ),
    schema =
      StructType(
        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
      ),
    input = PlainDataset(
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
      new VectorIndexer()
        .setInputCol("features")
        .setOutputCol("indexedFeatures")
        .setMaxCategories(4),
      new GBTRegressor()
        .setLabelCol("label")
        .setFeaturesCol("indexedFeatures")
        .setMaxIter(10)
    ),
    schema =
      StructType(
        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
      ),
    input = PlainDataset(
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
      new KMeans().setK(2).setSeed(1L)
    ),
    schema =
      StructType(
        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
      ),
    input = PlainDataset(
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
      new GaussianMixture().setK(2)
    ),
    schema =
      StructType(
        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
      ),
    input = PlainDataset(
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

  modelTest(
    trainData = session.createDataFrame(Seq(
      (0, "Hi I heard about Spark"),
      (1, "I wish Java could use case classes"),
      (2, "Logistic,regression,models,are,neat")
    )).toDF("id", "sentence"),
    stages = Seq(
      new RegexTokenizer()
        .setInputCol("sentence")
        .setOutputCol("words")
        .setPattern("\\W")
    ),
    schema =
      StructType(
        StructField("sentence", StringType) :: Nil
      ),
    input = PlainDataset(
      Column("sentence", Seq(
        "Hi I heard about Spark",
        "I wish Java could use case classes",
        "Logistic,regression,models,are,neat"
      ))
    )
  )

  modelTest(
    trainData = session.createDataFrame(Seq(
      (0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0)
    )).toDF("id", "hour", "mobile", "userFeatures", "clicked"),
    stages = Seq(
      new VectorAssembler()
        .setInputCols(Array("hour", "mobile", "userFeatures"))
        .setOutputCol("features")
    ),
    schema =
      StructType(
        StructField("hour", IntegerType) ::
        StructField("mobile", DoubleType) ::
        StructField("userFeatures", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) ::
        Nil
      ),
    input = PlainDataset(
      Column("hour", Seq(18)),
      Column("mobile", Seq(1.0)),
      Column("userFeatures", Seq(Vectors.dense(0.0, 10.0, 0.5)))
    ),
    sample = Some(
      session.createDataFrame(Seq(
        (0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0)
      )).toDF("id", "hour", "mobile", "userFeatures", "clicked")
    )
  )

  modelTest(
    trainData = session.read.format("libsvm").load(getClass.getClassLoader.getResource("sample_lda_libsvm_data.txt").getPath),
    stages = Seq(
      new LDA().setK(10).setMaxIter(10)
    ),
    schema =
      StructType(
        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
      ),
    input = {
        val df = session.read.format("libsvm").load(getClass.getClassLoader.getResource("sample_lda_libsvm_data.txt").getPath)
        PlainDataset.fromDataFrame(df.select("features"))
      },
    compare = (a: PlainDataset, b: PlainDataset) => {

      def compareDistribution(): Boolean = {
        val i1 = a.columnByName("topicDistribution").items
        val i2 = b.columnByName("topicDistribution").items

        (0 until a.size).map(i => {
          val v1 = i1(i).asInstanceOf[DenseVector]
          val v2 = i2(i).asInstanceOf[DenseVector]
          v1.values.zip(v2.values).map({case (d1, d2) => if ((d1 - d2).abs < 1) 1 else 0}).product
        }).product == 1
      }

      (a.columnNames.sorted == b.columnNames.sorted) &&
        (a.columnByName("features") == b.columnByName("features")) &&
        compareDistribution()
    }
  )


  def modelTest(
    trainData: DataFrame,
    stages: Seq[PipelineStage],
    schema: StructType,
    input: PlainDataset,
    sample: Option[DataFrame] = None,
    compare: (PlainDataset, PlainDataset) => Boolean = (a: PlainDataset, b: PlainDataset) => a.equals(b)
  ): Unit = {
    val name = stages.map(_.getClass.getSimpleName).foldLeft("") {
      case ("", b) => b
      case (a, b) => a + "-" + b
    }

    it(name) {
      val pipeline = new Pipeline().setStages(stages.toArray)
      val pipelineModel = pipeline.fit(trainData)


      val interpSample = sample.getOrElse(session.createDataFrame(session.sparkContext.emptyRDD[Row], schema))
      val transformer = FastInterpreter.fromTransformer(pipelineModel, interpSample)

      val out = transformer(input)

      val origDf = pipelineModel.transform(input.toDataFrame(session, schema))
      val origToPlain = PlainDataset.fromDataFrame(origDf)

      def mkMessage(info: String)(local: PlainDataset, default: PlainDataset): String = {
        info + "\n" +
          "Local:\n" +
          local.toString + "\n" +
          "Spark:\n" +
          default.toString + "\n"
      }

      try {
        if (!compare(out, origToPlain)) {
          fail(mkMessage("Got different outputs:")(out, origToPlain))
        }
      } catch {
        case e: Throwable =>
          fail(mkMessage("Compare function failed")(out, origToPlain), e)
      }

    }
  }
}
