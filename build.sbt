ThisBuild / scalaVersion  := "2.11.8"
ThisBuild / version  := "0.0.1"
ThisBuild / organization := "io.hydrosphere"

lazy val sparkVersionKey = settingKey[String]("Spark version")

ThisBuild / sparkVersionKey := sys.props.getOrElse("sparkVersion", "2.3.0")

lazy val fastserving = project.in(file("fastserving"))
  .settings(PublishSettings.settings: _*)
  .settings(
    name := "fastserving",
    libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % "test",
    unmanagedSourceDirectories in Compile += {
      val sparkVersion = sparkVersionKey.value
      val dir = if (sparkVersion == "2.3.0") "2_3_0" else "2_x_x"
      baseDirectory.value / "src" / "main" / s"spark_$dir"
    },
    version := version.value + s"_spark-${sparkVersionKey.value}",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % sparkVersionKey.value,
      "org.apache.spark" %% "spark-sql" % sparkVersionKey.value,
      "org.apache.spark" %% "spark-mllib" % sparkVersionKey.value
    )
  )

lazy val bench = project.in(file("bench"))
  .dependsOn(fastserving % "compile->compile;compile->test")
  .enablePlugins(JmhPlugin)
  .settings(
    sourceGenerators in Compile += sourceManaged.map(out => BenchGen.genAll(out)).taskValue,
    skip in publish := true
  )

lazy val root = project.in(file(".")).aggregate(fastserving)
  .settings(
    skip in publish := true
  )
