package fastserving

import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{ChiSqSelector, StringIndexer, VectorIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}
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

case class ModelSetup(
  stages: Seq[PipelineStage],
  trainSample: Sample.RealSample,
  interpSample: Sample,
  input: PlainDataset
)

object TestSetups {

  val VectorUDT = ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType

  val DenseFeaturesDoubleLabels = Seq(
    (Vectors.dense(4.0, 0.2, 3.0, 4.0, 5.0), 1.0),
    (Vectors.dense(3.0, 0.3, 1.0, 4.1, 5.0), 1.0),
    (Vectors.dense(2.0, 0.5, 3.2, 4.0, 5.0), 1.0),
    (Vectors.dense(5.0, 0.7, 1.5, 4.0, 5.0), 1.0),
    (Vectors.dense(1.0, 0.1, 7.0, 4.0, 5.0), 0.0),
    (Vectors.dense(8.0, 0.3, 5.0, 1.0, 7.0), 0.0)
  )

  val `StrInd-VecInd-RandForestClassifier` = ModelSetup(
    stages = Seq(
        new StringIndexer().setInputCol("label").setOutputCol("indexedLabel"),
        new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4),
        new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)
      ),
    trainSample = Sample.real(_.createDataFrame(DenseFeaturesDoubleLabels).toDF("features", "label")),
    interpSample = Sample.empty(StructType(
        StructField("features", VectorUDT) :: Nil
    )),
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

  val `ChiSqSelector` = {
    val train = Seq(
      (7, Vectors.dense(0.0, 0.0, 18.0, 1.0), 1.0),
      (8, Vectors.dense(0.0, 1.0, 12.0, 0.0), 0.0),
      (9, Vectors.dense(1.0, 0.0, 15.0, 0.1), 0.0)
    )
    ModelSetup(
      stages = Seq(
        new ChiSqSelector().setNumTopFeatures(1).setFeaturesCol("features").setLabelCol("clicked").setOutputCol("selectedFeatures")
      ),
      trainSample = Sample.real(_.createDataFrame(train).toDF("id", "features", "clicked")),
      interpSample = Sample.empty(StructType(
        StructField("features", VectorUDT) :: Nil
      )),
      input = PlainDataset(
        Column("features", Seq(
          Vectors.dense(0.0, 0.0, 18.0, 1.0),
          Vectors.dense(0.0, 1.0, 12.0, 0.0),
          Vectors.dense(1.0, 0.0, 15.0, 0.1)
        ))
      )
    )
  }

  val `Tokenizer-HashingTF-IDF` = {
    val train = Seq(
      (0.0, "Hi I heard about Spark"),
      (0.0, "I wish Java could use case classes"),
      (1.0, "Logistic regression models are neat")
    )
    ModelSetup(
      stages = Seq(
        new Tokenizer().setInputCol("sentence").setOutputCol("words"),
        new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20),
        new IDF().setInputCol("rawFeatures").setOutputCol("features")
      ),
      trainSample = Sample.real(_.createDataFrame(train).toDF("label", "sentence")),
      interpSample = Sample.empty(StructType(
        StructField("sentence", StringType) :: Nil
      )),
      input = PlainDataset(
        Column("sentence", Seq(
          "Hi I heard about Spark",
          "I wish Java could use case classes",
          "Logistic regression models are neat"
        ))
      )
    )
  }

  val `NGram` = {
    val train = Seq(
      (0, Array("Hydrosphere", "is", "such", "a", "cool", "company")),
      (1, Array("Big", "data", "rules", "the", "world")),
      (2, Array("Cloud", "solutions", "are", "our", "future"))
    )
    ModelSetup(
      stages = Seq(
        new NGram().setN(2).setInputCol("words").setOutputCol("ngrams")
      ),
      trainSample = Sample.real(_.createDataFrame(train).toDF("id", "words")),
      interpSample = Sample.empty(StructType(
        StructField("words", new ArrayType(StringType, false)) :: Nil
      )),
      input = PlainDataset(
        Column("words", Seq(
          Seq("Provectus", "is", "such", "a", "cool", "company"),
          Seq("Big", "data", "rules", "the", "world"),
          Seq("Cloud", "solutions", "are", "our", "future")
        ))
      )
    )
  }

  val `StandardScaler` = {
    val train = Seq(
      Vectors.dense(0.0, 10.3, 1.0, 4.0, 5.0),
      Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
      Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
    ).map(Tuple1.apply)
    ModelSetup(
      stages = Seq(
        new StandardScaler()
          .setInputCol("features")
          .setOutputCol("scaledFeatures")
          .setWithStd(true)
          .setWithMean(false)
      ),
      trainSample = Sample.real(_.createDataFrame(train).toDF("features")),
      interpSample = Sample.empty(StructType(
        StructField("features", VectorUDT) :: Nil
      )),
      input = PlainDataset(
        Column("features", Seq(
          Vectors.dense(0.0, 10.3, 1.0, 4.0, 5.0),
          Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
          Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
        ))
      )
    )
  }

  val `StopWordRemover` = {
    val train = Seq(
      (0, Seq("I", "saw", "the", "red", "balloon")),
      (1, Seq("Mary", "had", "a", "little", "lamb"))
    )
    ModelSetup(
      stages = Seq(
        new StopWordsRemover()
          .setInputCol("raw")
          .setOutputCol("filtered")
      ),
      trainSample = Sample.real(_.createDataFrame(train).toDF("id", "raw")),
      interpSample = Sample.empty(StructType(
        StructField("raw", new ArrayType(StringType, false)) :: Nil
      )),
      input = PlainDataset(
        Column("raw", Seq(
          Seq("I", "saw", "the", "red", "balloon"),
          Seq("Mary", "had", "a", "little", "lamb")
        ))
      )
    )
  }

  val `MaxAbsScaler` = {
    val train = Seq(
      (0, Vectors.dense(1.0, 0.1, -8.0)),
      (1, Vectors.dense(2.0, 1.0, -4.0)),
      (2, Vectors.dense(4.0, 10.0, 8.0))
    )
    ModelSetup(
      stages = Seq(
        new MaxAbsScaler()
          .setInputCol("features")
          .setOutputCol("scaledFeatures")
      ),
      trainSample = Sample.real(_.createDataFrame(train).toDF("id", "features")),
      interpSample = Sample.empty(StructType(
        StructField("features", VectorUDT) :: Nil
      )),
      input = PlainDataset(
        Column("features", Seq(
          Vectors.dense(1.0, 0.1, -8.0),
          Vectors.dense(2.0, 1.0, -4.0),
          Vectors.dense(4.0, 10.0, 8.0)
        ))
      )
    )
  }

  val `MinMaxScaler` = {
    val train = Seq(
      (0, Vectors.dense(1.0, 0.1, -1.0)),
      (1, Vectors.dense(2.0, 1.1, 1.0)),
      (2, Vectors.dense(3.0, 10.1, 3.0))
    )
    ModelSetup(
      stages = Seq(
        new MinMaxScaler()
          .setInputCol("features")
          .setOutputCol("scaledFeatures")
      ),
      trainSample = Sample.real(_.createDataFrame(train).toDF("id", "features")),
      interpSample = Sample.empty(StructType(
        StructField("features", VectorUDT) :: Nil
      )),
      input = PlainDataset(
        Column("features", Seq(
          Vectors.dense(1.0, 0.1, -1.0),
          Vectors.dense(2.0, 1.1, 1.0),
          Vectors.dense(3.0, 10.1, 3.0)
        ))
      )
    )
  }

  val `StringIndexer-OneHotEncoder` = {
    val train = Seq(
      (0, "a"), (1, "b"), (2, "c"),
      (3, "a"), (4, "a"), (5, "c")
    )
    ModelSetup(
      stages = Seq(
        new StringIndexer()
          .setInputCol("category")
          .setOutputCol("categoryIndex"),
        new OneHotEncoder()
          .setInputCol("categoryIndex")
          .setOutputCol("categoryVec")
      ),
      trainSample = Sample.real(_.createDataFrame(train).toDF("id", "category")),
      interpSample = Sample.empty(StructType(
        StructField("category", StringType) :: Nil
      )),
      input = PlainDataset(
        Column("category", Seq(
          "a", "b", "c", "a", "a", "c"
        ))
      )
    )
  }

  val `PCA` = {
    val train = Seq(
      Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
      Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
      Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
    ).map(Tuple1.apply)
    ModelSetup(
      stages = Seq(
        new PCA()
          .setInputCol("features")
          .setOutputCol("pcaFeatures")
          .setK(3)
      ),
      trainSample = Sample.real(_.createDataFrame(train).toDF("features")),
      interpSample = Sample.empty(StructType(
        StructField("features", VectorUDT) :: Nil
      )),
      input = PlainDataset(
        Column("features", Seq(
          Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
          Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
          Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
        ))
      )
    )
  }

  val `Normalizer` = {
    val train = Seq(
      (0, Vectors.dense(1.0, 0.5, -1.0)),
      (1, Vectors.dense(2.0, 1.0, 1.0)),
      (2, Vectors.dense(4.0, 10.0, 2.0))
    )
    ModelSetup(
      stages = Seq(
        new Normalizer()
          .setInputCol("features")
          .setOutputCol("normFeatures")
          .setP(1.0)
      ),
      trainSample = Sample.real(_.createDataFrame(train).toDF("id", "features")),
      interpSample = Sample.empty(StructType(
        StructField("features", VectorUDT) :: Nil
      )),
      input = PlainDataset(
        Column("features", Seq(
          Vectors.dense(1.0, 0.5, -1.0),
          Vectors.dense(2.0, 1.0, 1.0),
          Vectors.dense(4.0, 10.0, 2.0)
        ))
      )
    )
  }

  val `DCT` = {
    val train = Seq(
      Vectors.dense(0.0, 1.0, -2.0, 3.0),
      Vectors.dense(-1.0, 2.0, 4.0, -7.0),
      Vectors.dense(14.0, -2.0, -5.0, 1.0)
    ).map(Tuple1.apply)
    ModelSetup(
      stages = Seq(
        new DCT()
          .setInputCol("features")
          .setOutputCol("featuresDCT")
          .setInverse(false)
      ),
      trainSample = Sample.real(_.createDataFrame(train).toDF("features")),
      interpSample = Sample.empty(StructType(
        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
      )),
      input = PlainDataset(
        Column("features", Seq(
          Vectors.dense(0.0, 1.0, -2.0, 3.0),
          Vectors.dense(-1.0, 2.0, 4.0, -7.0),
          Vectors.dense(14.0, -2.0, -5.0, 1.0)
        ))
      )
    )
  }

  val `NaiveBayes` = {
    ModelSetup(
      stages = Seq(
        new NaiveBayes()
      ),
      trainSample = Sample.real(_.createDataFrame(DenseFeaturesDoubleLabels).toDF("features", "label")),
      interpSample = Sample.empty(StructType(
        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
      )),
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
  }

  val `Binarizer` = {
    val train = Seq(
      (0, 0.1),
      (1, 0.8),
      (2, 0.2)
    )
    ModelSetup(
      stages = Seq(
        new Binarizer()
          .setInputCol("feature")
          .setOutputCol("binarized_feature")
          .setThreshold(5.0)
      ),
      trainSample = Sample.real(_.createDataFrame(train).toDF("id", "feature")),
      interpSample = Sample.empty(StructType(
        StructField("feature", DoubleType) :: Nil
      )),
      input = PlainDataset(
        Column("feature", Seq(0.1, 0.8, 0.2))
      )
    )
  }

  val `StringInd-VectorInd-GBTCLassifier` = {
    ModelSetup(
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
      trainSample = Sample.real(_.createDataFrame(DenseFeaturesDoubleLabels).toDF("features", "label")),
      interpSample = Sample.empty(StructType(
        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
      )),
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
  }

  val `StringInd-VectorInd-DecisionTreeClassifier` = {
    ModelSetup(
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
      trainSample = Sample.real(_.createDataFrame(DenseFeaturesDoubleLabels).toDF("features", "label")),
      interpSample = Sample.empty(StructType(
        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
      )),
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
  }

  val `Tokenizer-HashingTF-LinearRegression` = {
    val train = Seq(
      (0L, "a b c d e spark", 1.0),
      (1L, "b d", 0.0),
      (2L, "spark f g h", 1.0),
      (3L, "hadoop mapreduce", 0.0)
    )
    ModelSetup(
      stages = Seq(
        new Tokenizer().setInputCol("text").setOutputCol("words"),
        new HashingTF().setNumFeatures(1000).setInputCol("words").setOutputCol("features"),
        new LinearRegression()
          .setMaxIter(10)
          .setRegParam(0.3)
          .setElasticNetParam(0.8)
      ),
      trainSample = Sample.real(_.createDataFrame(train).toDF("id", "text", "label")),
      interpSample = Sample.empty(StructType(
        StructField("text", StringType) :: Nil
      )),
      input = PlainDataset(
        Column("text", Seq(
          "a b c d e spark",
          "b d",
          "spark f g h",
          "hadoop mapreduce"
        ))
      )
    )
  }

  val `VectorInd-DecisionTreeRegr` = {
    ModelSetup(
      stages = Seq(
        new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4),
        new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures")
      ),
      trainSample = Sample.real(_.createDataFrame(DenseFeaturesDoubleLabels).toDF("features", "label")),
      interpSample = Sample.empty(StructType(
        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
      )),
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
  }

  val `VectorInd-RandomForestRegr` = {
    ModelSetup(
      stages = Seq(
        new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4),
        new RandomForestRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures")
      ),
      trainSample = Sample.real(_.createDataFrame(DenseFeaturesDoubleLabels).toDF("features", "label")),
      interpSample = Sample.empty(StructType(
        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
      )),
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
  }

  val `VectorInd-GBTRegr` = {
    ModelSetup(
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
      trainSample = Sample.real(_.createDataFrame(DenseFeaturesDoubleLabels).toDF("features", "label")),
      interpSample = Sample.empty(StructType(
        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
      )),
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
  }

  val `KMeans` = {
    ModelSetup(
      stages = Seq(
        new KMeans().setK(2).setSeed(1L)
      ),
      trainSample = Sample.real(_.createDataFrame(DenseFeaturesDoubleLabels).toDF("features", "label")),
      interpSample = Sample.empty(StructType(
        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
      )),
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
  }

  val `GaussianMixture` = {
    ModelSetup(
      stages = Seq(
        new GaussianMixture().setK(2)
      ),
      trainSample = Sample.real(_.createDataFrame(DenseFeaturesDoubleLabels).toDF("features", "label")),
      interpSample = Sample.empty(StructType(
        StructField("features", ScalaReflection.schemaFor[org.apache.spark.ml.linalg.Vector].dataType) :: Nil
      )),
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
  }

  val `RegexTokenizer` = {
    val train = Seq(
      (0, "Hi I heard about Spark"),
      (1, "I wish Java could use case classes"),
      (2, "Logistic,regression,models,are,neat")
    )
    ModelSetup(
      stages = Seq(
        new RegexTokenizer()
          .setInputCol("sentence")
          .setOutputCol("words")
          .setPattern("\\W")
      ),
      trainSample = Sample.real(_.createDataFrame(train).toDF("id", "sentence")),
      interpSample = Sample.empty(StructType(
        StructField("sentence", StringType) :: Nil
      )),
      input = PlainDataset(
        Column("sentence", Seq(
          "Hi I heard about Spark",
          "I wish Java could use case classes",
          "Logistic,regression,models,are,neat"
        ))
      )
    )
  }

  val `VectorAssembler` = {
    val train = Seq(
      (0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0)
    )
    val sample = Seq(
      (18, 1.0, Vectors.dense(0.0, 10.0, 0.5))
    )
    ModelSetup(
      stages = Seq(
        new VectorAssembler()
          .setInputCols(Array("hour", "mobile", "userFeatures"))
          .setOutputCol("features")
      ),
      trainSample = Sample.real(_.createDataFrame(train).toDF("id", "hour", "mobile", "userFeatures", "clicked")),
      interpSample = Sample.real(_.createDataFrame(sample).toDF("hour", "mobile", "userFeatures")),
      input = PlainDataset(
        Column("hour", Seq(18)),
        Column("mobile", Seq(1.0)),
        Column("userFeatures", Seq(Vectors.dense(0.0, 10.0, 0.5)))
      )
    )
  }

  val `LDA` = {
    val trainSample = Sample.real(sess => {
      val path = getClass.getClassLoader.getResource("sample_lda_libsvm_data.txt").getPath
      sess.read.format("libsvm").load(path)
    })
    val interpSample = Sample.real(sess => {
      val path = getClass.getClassLoader.getResource("sample_lda_libsvm_data.txt").getPath
      sess.read.format("libsvm").load(path).select("features")
    })

    ModelSetup(
      stages = Seq(
        new LDA().setK(10).setMaxIter(10)
      ),
      trainSample = trainSample,
      interpSample = Sample.empty(StructType(
        StructField("features", VectorUDT) :: Nil
      )),
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
  }

}
