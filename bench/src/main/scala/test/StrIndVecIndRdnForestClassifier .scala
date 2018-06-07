package test

import java.util.concurrent.TimeUnit

import localserve._
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.openjdk.jmh.annotations._
import org.openjdk.jmh.annotations.Benchmark

@State(Scope.Benchmark)
@Measurement(timeUnit = TimeUnit.MILLISECONDS)
class StrIndVecIndRdnForestClassifier {

  val setup = BenchSetup(TestSetups.`StrInd-VecInd-RandForestClassifier`)

  val fastTransformer = setup.fastTransformer
  val plainDs = setup.setup.input

  val transfromer = setup.pipelineModel
  val sparkDf = setup.sparkDf

  @Benchmark
  def fast(): PlainDataset = {
    fastTransformer(plainDs)
  }

  @Benchmark
  def spark(): Array[Row] = {
    transfromer.transform(sparkDf).collect()
  }


}

object BenchAll {

//  val conf = new SparkConf()
//    .setMaster("local[2]")
//    .setAppName("test")
//    .set("spark.ui.enabled", "false")
//
//  val session: SparkSession = SparkSession.builder().config(conf).getOrCreate()
//
//  class Bench(val setup: ModelSetup) {
//    import setup._
//
//    val model = {
//      val pipeline = new Pipeline().setStages(setup.stages.toArray)
//      val trainDf = setup.trainSample.mkDf(session)
//      pipeline.fit(trainDf)
//    }
//
//    val transformer = {
//      import setup._
//
//      val sampleDf = setup.interpSample.mkDf(session)
//      FastInterpreter.fromTransformer(model, sampleDf)
//    }
//
//    val vanillaDf = {
//      val schema = setup.interpSample.mkDf(session).schema
//      setup.input.toDataFrame(session, schema)
//    }
//  }
//
//  import TestSetups._
//
//  def bench(setup: ModelSetup): ModelSetup = new Bench(setup)
//
//  val strIndVecIndRandForestClassfier = bench(`StrInd-VecInd-RandForestClassifier`)
//  val chisqlSelector = bench(`ChiSqSelector`)
//  val tokenizerHashingTFIdf = bench(`Tokenizer-HashingTF-IDF`)
//  val ngram = bench(`NGram`)
//  val standardScaler = bench(`StandardScaler`)
//  val stopWordRemover = bench(`StopWordRemover`)
//  val maxAbsScaler = bench(`MaxAbsScaler`)
//  val minMaxScaler = bench(`MinMaxScaler`)
//  val stringIndexerOneHotEncoder = bench(`StringIndexer-OneHotEncoder`)
//  val pca = bench(`PCA`)
//  val normalizer = bench(`Normalizer`)
//  val dct = bench(`DCT`)
//  val naiveBayes = bench(`NaiveBayes`)
//  val binarizer = bench(`Binarizer`)
//  val stringIndVectorIndGBT = bench(`StringInd-VectorInd-GBTCLassifier`)
//  val stringIndVectorIndDecisionTreeClassifier = bench(`StringInd-VectorInd-DecisionTreeClassifier`)
//  val tokenizerHashingLinearRegr = bench(`Tokenizer-HashingTF-LinearRegression`)
//  val vectorIndDecisionTreeRegr = bench(`VectorInd-DecisionTreeRegr`)
//  val vectorIndRandomForestRegr = bench(`VectorInd-RandomForestRegr`)
//  val vectorIndGBTRegr = bench(`VectorInd-GBTRegr`)
//  val kmeans = bench(`KMeans`)
//  val gaussianMixture = bench(`GaussianMixture`)
//  val regexTokenizer = bench(`RegexTokenizer`)
//  //val vectorAssembler = bench(`VectorAssembler`)
//  val lda = bench(`LDA`, LDAcompare)

}
