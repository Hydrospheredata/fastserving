package fastserve

import org.apache.spark.SparkConf
import org.apache.spark.ml.{Pipeline, Transformer}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{StringIndexer, VectorIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.analysis.UnresolvedAttribute
import org.apache.spark.sql.catalyst.expressions.{Alias, AttributeReference, ScalaUDF}
import org.apache.spark.sql.catalyst.plans.logical.{AnalysisBarrier, LogicalPlan, Project, ReturnAnswer}
import org.apache.spark.sql.execution.LogicalRDD

import scala.collection.mutable.ArrayBuffer

object Main extends App {

  println("YOYOYO")

  val conf = new SparkConf()
    .setMaster("local[2]")
    .setAppName("test")
    .set("spark.ui.enabled", "false")

  val session: SparkSession = SparkSession.builder().config(conf).getOrCreate()

  val test = new CatalystTest(session)

  val emptyDf = session.createDataFrame(session.sparkContext.emptyRDD[Row], test.inputData.schema)
  val logicalPlan = test.pipelineModel.transform(emptyDf).queryExecution.logical

//  try {
//    val transformer = LocalTransformer.fromLogicalPlan(logicalPlan)
//    val input = LocalData(
//      columnsId = Map("features" -> 0),
//      columns = Seq(
//        LocalColumn("features", Seq(
//          Vectors.dense(4.0, 0.2, 3.0, 4.0, 5.0),
//          Vectors.dense(3.0, 0.3, 1.0, 4.1, 5.0),
//          Vectors.dense(2.0, 0.5, 3.2, 4.0, 5.0),
//          Vectors.dense(5.0, 0.7, 1.5, 4.0, 5.0),
//          Vectors.dense(1.0, 0.1, 7.0, 4.0, 5.0),
//          Vectors.dense(8.0, 0.3, 5.0, 1.0, 7.0)
//        ))
//      )
//    )
//    val out = transformer.transform(input)
//    println(out)
//  } catch {
//    case e: Throwable => e.printStackTrace()
//  } finally {
//    session.close()
//  }
}

class CatalystTest(session: SparkSession) {

  def modelPath(modelName: String): String = s"./target/test_models/${session.version}/$modelName"

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
}
