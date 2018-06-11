package test

import fastserving.iterpr.FastInterpreter
import fastserving.{FastTransformer, ModelSetup}
import org.apache.spark.SparkConf
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

case class BenchSetup(
  session: SparkSession,
  setup: ModelSetup,
  pipelineModel: PipelineModel,
  fastTransformer: FastTransformer,
  sparkDf: DataFrame
)

object BenchSetup {

  def apply(setup: ModelSetup): BenchSetup = {
    val session = {
      val conf = new SparkConf()
        .setMaster("local[2]")
        .setAppName("test")
        .set("spark.ui.enabled", "false")

      SparkSession.builder().config(conf).getOrCreate()
    }

    val model = {
      val pipeline = new Pipeline().setStages(setup.stages.toArray)
      val trainDf = setup.trainSample.mkDf(session)
      pipeline.fit(trainDf)
    }

    val transformer = {
      val sampleDf = setup.interpSample.mkDf(session)
      FastInterpreter.fromTransformer(model, sampleDf)
    }

    val sparkDf = {
      val schema = setup.interpSample.mkDf(session).schema
      setup.input.toDataFrame(session, schema)
    }
    BenchSetup(session, setup, model, transformer, sparkDf)
  }
}
