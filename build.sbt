ThisBuild / scalaVersion  := "2.11.8"

val serve = project.in(file("serve"))
  .settings(
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % "2.3.0",
      "org.apache.spark" %% "spark-sql" % "2.3.0",
      "org.apache.spark" %% "spark-mllib" % "2.3.0"
    )
  )

val bench = project.in(file("bench"))
  .dependsOn(serve)
  .enablePlugins(JmhPlugin)
