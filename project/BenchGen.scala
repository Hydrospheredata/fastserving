import sbt._

import sbt.io.IO

object BenchGen {

  val Names = Seq(
    "`StrInd-VecInd-RandForestClassifier`",
    "`ChiSqSelector`",
    "`Tokenizer-HashingTF-IDF`",
    "`NGram`",
    "`StandardScaler`",
    "`StopWordRemover`",
    "`MaxAbsScaler`",
    "`MinMaxScaler`",
    "`StringIndexer-OneHotEncoder`",
    "`PCA`",
    "`Normalizer`",
    "`DCT`",
    "`NaiveBayes`",
    "`Binarizer`",
    "`StringInd-VectorInd-GBTCLassifier`",
    "`StringInd-VectorInd-DecisionTreeClassifier`",
    "`Tokenizer-HashingTF-LinearRegression`",
    "`VectorInd-DecisionTreeRegr`",
    "`VectorInd-RandomForestRegr`",
    "`VectorInd-GBTRegr`",
    "`KMeans`",
    "`RegexTokenizer`",
    "`VectorAssembler`",
    "`LDA`"
  )

  def genAll(out: File): Seq[File] = Names.map(n => genOne(n, out))

  def genOne(name: String, out: File): File = {
    val className = name.replaceAll("-", "_").replaceAll("`", "")
    val target = out / "test" /  (className + ".scala")
    val data = template(className, name)
    IO.write(target, data)
    target
  }

  def template(className: String, name: String): String = {
//    val className = name.replaceAll("-", "_")
    s"""
       |package test
       |
       |import java.util.concurrent.TimeUnit
       |
       |import fastserving._
       |import org.apache.spark.SparkConf
       |import org.apache.spark.ml.Pipeline
       |import org.apache.spark.sql.{DataFrame, Row, SparkSession}
       |import org.openjdk.jmh.annotations._
       |import org.openjdk.jmh.annotations.Benchmark
       |
       |@State(Scope.Benchmark)
       |@Measurement(timeUnit = TimeUnit.MILLISECONDS)
       |class ${className} {
       |
       |  val setup = BenchSetup(TestSetups.${name})
       |  val fastTransformer = setup.fastTransformer
       |  val plainDs = setup.setup.input
       |  val transfromer = setup.pipelineModel
       |  val sparkDf = setup.sparkDf
       |
       |  @Benchmark
       |  def fast(): PlainDataset = {
       |    fastTransformer(plainDs)
       |  }
       |
       |  @Benchmark
       |  def spark(): Array[Row] = {
       |    transfromer.transform(sparkDf).collect()
       |  }
       |}
     """.stripMargin
  }

}
