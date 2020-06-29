package org.com.jonas

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession}
import breeze.linalg.{DenseMatrix, DenseVector, normalize}

object Validate {
  /**
    * @param args
    * args(0): Path of sample
    * args(1): Path of model
    * args(2): Path to result
    *
    * @return result
    * file with structure: M;k;finalPi,finalA,finalB
    */
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    val sparkSession = SparkSession.builder.appName("Spark-HMM").getOrCreate()

    import scala.io.Source
    val arraymodel = Source.fromFile(args(1)).mkString.split(";")

    val M = arraymodel(0).toInt
    val k = arraymodel(1).toInt
    val T = 1000

    val Pi: DenseVector[Double] = new DenseVector(arraymodel(2).split(",").map(_.toDouble))
    val A = new DenseMatrix(M, M, arraymodel(3).split(",").map(_.toDouble))
    val B = new DenseMatrix(M, k, arraymodel(4).split(",").map(_.toDouble))

    val sample: DataFrame = sparkSession.read.csv(args(0))
      .withColumnRenamed("_c0", "workitem").withColumnRenamed("_c1", "str_obs")

    val result = hmm.BaumWelchAlgorithm.validate(sample, M, k, T, Pi, A, B)

    result
      .select("workitem", "prob").coalesce(1)
      .write.format("com.databricks.spark.csv").save(args(2))

    println("*****************************************************************************************")

    sparkSession.stop()
  }

}
