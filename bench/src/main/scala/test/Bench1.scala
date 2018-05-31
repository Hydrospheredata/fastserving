package test

import fastserve._
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{StringIndexer, VectorIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{Row, SparkSession}
import org.openjdk.jmh.annotations.Benchmark

import scala.collection.mutable.ArrayBuffer

class Bench1 {

  import Bench1._

//  @Benchmark
//  def sparkml(): Unit = {
//    pipelineModel.transform(inputData).collect()
//  }

  @Benchmark
  def local(): Unit = {
    transformer(input)
  }


//  @Benchmark
//  def compose(): Unit = {
//    y(4)
//  }
//  @Benchmark
//  def direct(): Unit = {
//    x(4)
//  }
}

object Bench1 {
  val x = (i: Int) => i
  val y = (identity[Int] _).compose(x)

  val conf = new SparkConf()
    .setMaster("local[2]")
    .setAppName("test")
    .set("spark.ui.enabled", "false")

  val session: SparkSession = SparkSession.builder().config(conf).getOrCreate()

  val steps = Seq(
    new StringIndexer().setInputCol("label").setOutputCol("indexedLabel"),
    new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4),
    new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)
  )
  val data = session.createDataFrame(Seq(
    (Vectors.dense(4.0, 0.2, 3.0, 4.0, 5.0), 1.0),
    (Vectors.dense(3.0, 0.3, 1.0, 4.1, 5.0), 1.0),
    (Vectors.dense(2.0, 0.5, 3.2, 4.0, 5.0), 1.0),
    (Vectors.dense(5.0, 0.7, 1.5, 4.0, 5.0), 1.0),
    (Vectors.dense(1.0, 0.1, 7.0, 4.0, 5.0), 0.0),
    (Vectors.dense(8.0, 0.3, 5.0, 1.0, 7.0), 0.0)
  )).toDF("features", "label")

  val inputData = data.drop(data.col("label"))

  val pipeline = new Pipeline().setStages(steps.toArray)
  val pipelineModel = pipeline.fit(data)


  val emptyDf = session.createDataFrame(session.sparkContext.emptyRDD[Row], inputData.schema)
  val transformer = LogicalPlanInterpreter.fromTransformer(pipelineModel, emptyDf)

  val input = PlainDataset(
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
}
