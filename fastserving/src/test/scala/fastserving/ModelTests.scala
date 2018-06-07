package fastserving

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterAll, FunSpec}

class ModelTests extends FunSpec with BeforeAndAfterAll {

  case class Test(
    setup: ModelSetup,
    compare: (PlainDataset, PlainDataset) => Boolean = (a: PlainDataset, b: PlainDataset) => a.equals(b)
  )

  import TestSetups._

  val LDAcompare = (a: PlainDataset, b: PlainDataset) => {

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

  val testingModels: Seq[Test] = Seq(
    Test(`StrInd-VecInd-RandForestClassifier`),
    Test(`ChiSqSelector`),
    Test(`Tokenizer-HashingTF-IDF`),
    Test(`NGram`),
    Test(`StandardScaler`),
    Test(`StopWordRemover`),
    Test(`MaxAbsScaler`),
    Test(`MinMaxScaler`),
    Test(`StringIndexer-OneHotEncoder`),
    Test(`PCA`),
    Test(`Normalizer`),
    Test(`DCT`),
    Test(`NaiveBayes`),
    Test(`Binarizer`),
    Test(`StringInd-VectorInd-GBTCLassifier`),
    Test(`StringInd-VectorInd-DecisionTreeClassifier`),
    Test(`Tokenizer-HashingTF-LinearRegression`),
    Test(`VectorInd-DecisionTreeRegr`),
    Test(`VectorInd-RandomForestRegr`),
    Test(`VectorInd-GBTRegr`),
    Test(`KMeans`),
    Test(`GaussianMixture`),
    Test(`RegexTokenizer`),
    Test(`VectorAssembler`),
    Test(`LDA`, LDAcompare)
  )

  val conf = new SparkConf()
    .setMaster("local[2]")
    .setAppName("test2")
    .set("spark.ui.enabled", "false")

  val session: SparkSession = SparkSession.builder().config(conf).getOrCreate()

  testingModels.foreach(t => {
    import t._
    val name = setup.stages.map(_.getClass.getSimpleName).foldLeft("") {
      case ("", b) => b
      case (a, b) => a + "-" + b
    }

    it(name) {
      val pipeline = new Pipeline().setStages(setup.stages.toArray)
      val trainDf = setup.trainSample.mkDf(session)
      val pipelineModel = pipeline.fit(trainDf)

      val sampleDf = setup.interpSample.mkDf(session)
      val transformer = try {
        FastTransformer.build(pipelineModel, session, setup.interpSample)
      } catch {
        case e: Exception => fail(s"failed ${sampleDf.queryExecution.logical}", e)
      }

      val out = transformer(setup.input)

      val schema = setup.interpSample.mkDf(session).schema
      val origDf = pipelineModel.transform(setup.input.toDataFrame(session, schema))
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
  })


  override def afterAll(): Unit = session.close()
}
