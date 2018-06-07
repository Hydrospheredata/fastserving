ThisBuild / scalaVersion  := "2.11.8"
ThisBuild / version  := "0.0.1"
ThisBuild / organization := "io.hydrosphere"

val fastserving = project.in(file("fastserving"))
  .settings(
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % "2.3.0",
      "org.apache.spark" %% "spark-sql" % "2.3.0",
      "org.apache.spark" %% "spark-mllib" % "2.3.0",

      "org.scalatest" %% "scalatest" % "3.0.1" % "test"
    )
  )

val bench = project.in(file("bench"))
  .dependsOn(fastserving % "compile->compile;compile->test")
  .enablePlugins(JmhPlugin)
