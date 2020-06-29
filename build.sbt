name := "Spark-HMM"

version := "0.1"

scalaVersion := "2.11.12"

val sparkVersion = "2.4.5"

resolvers ++= Seq(
  "apache-snapshots" at "http://repository.apache.org/snapshots/",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly()
)

dependencyOverrides ++= Set(
  "io.netty" % "netty" % "3.9.9.Final",
  "commons-net" % "commons-net" % "2.2",
  "com.google.guava" % "guava" % "11.0.2"
)