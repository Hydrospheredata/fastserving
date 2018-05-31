import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
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

  try {
    val transformer = LocalTransformer.fromLogicalPlan(logicalPlan)
    val input = LocalData(
      columns = Map("features" -> 0),
      rows = Seq(
        LocalRow(ArrayBuffer(Vectors.dense(4.0, 0.2, 3.0, 4.0, 5.0))),
        LocalRow(ArrayBuffer(Vectors.dense(3.0, 0.3, 1.0, 4.1, 5.0))),
        LocalRow(ArrayBuffer(Vectors.dense(2.0, 0.5, 3.2, 4.0, 5.0))),
        LocalRow(ArrayBuffer(Vectors.dense(5.0, 0.7, 1.5, 4.0, 5.0))),
        LocalRow(ArrayBuffer(Vectors.dense(1.0, 0.1, 7.0, 4.0, 5.0))),
        LocalRow(ArrayBuffer(Vectors.dense(8.0, 0.3, 5.0, 1.0, 7.0)))
      )
    )
    val out = transformer.transform(input)
    println(out)
  } catch {
    case e: Throwable => e.printStackTrace()
  } finally {
    session.close()
  }
}

case class LocalRow(items: ArrayBuffer[Any]) {
  def append(a: Any): LocalRow = {
    items.append(a)
    this
  }
}

case class LocalData(
  columns: Map[String, Int],
  rows: Seq[LocalRow]
) {

  def addColumn(name: String, values: Seq[Any]): LocalData = {
    val upd = (rows zip values).map({case (row, v) => row.append(v)})
    copy(columns = columns + (name -> columns.size), upd)
  }

}


trait LocalTransformer { self =>
  def transform(input: LocalData): LocalData

  def andThen(f: LocalData => LocalData): LocalTransformer = {
    new LocalTransformer {
      def transform(input: LocalData): LocalData = f(self.transform(input))
    }
  }

  def before(f: LocalData => LocalData): LocalTransformer = {
    new LocalTransformer {
      def transform(input: LocalData): LocalData = self.transform(f(input))
    }
  }
}

object LocalTransformer {

  def apply(f: LocalData => LocalData): LocalTransformer = new LocalTransformer {
    override def transform(input: LocalData): LocalData = f(input)
  }

  val id: LocalTransformer = LocalTransformer(identity)

  def fromLogicalPlan(plan: LogicalPlan): LocalTransformer =  fromLogicalPlan(id, plan)

  def fromLogicalPlan(curr: LocalTransformer, plan: LogicalPlan): LocalTransformer = plan match {
    case ReturnAnswer(root) => fromLogicalPlan(curr, root)
    case barr: AnalysisBarrier => fromLogicalPlan(curr, barr.child)
    case lRdd: LogicalRDD => curr
    case Project(exprs, child) =>
      val trasformations = exprs.collect{
        case alias @ Alias(child, name) =>
          child match {
            case s @ ScalaUDF(func, _, children, _, name, nullable, determ) =>
              val inputNames = children.map{
                case AttributeReference(name, _, _, _) => name
                case att: UnresolvedAttribute => att.name
              }

              val f = (d: LocalData) => {
                //TODO
                val indexes = inputNames.map(n => d.columns(n))
                val values = d.rows.map(row => {
                  val in = indexes.map(i => row.items(i))
                  Util.callUDF(func, in)
                })
                d.addColumn(alias.name, values)
              }
              Some(f)
            case x =>
              println(s"Alias $alias whith $x")
              None
          }
        case other =>
          println(s"Ignore projection expr: $other")
          None
      }
      val next = trasformations.flatten.foldLeft(curr){case (t, f) => t.before(f)}
      fromLogicalPlan(next, child)
  }
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
