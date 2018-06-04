ThisBuild / scalaVersion  := "2.11.8"

val fastserve = project.in(file("fastserve"))
  .settings(
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % "2.3.0",
      "org.apache.spark" %% "spark-sql" % "2.3.0",
      "org.apache.spark" %% "spark-mllib" % "2.3.0",

      "org.scalatest" %% "scalatest" % "3.0.1" % "test"
    )
  )

val bench = project.in(file("bench"))
  .dependsOn(fastserve)
  .enablePlugins(JmhPlugin)
