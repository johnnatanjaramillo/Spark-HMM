package org.com.jonas

import java.io.FileInputStream

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession}
import breeze.linalg.{DenseMatrix, DenseVector, normalize}

object Train {
  /**
    * @param args
    * args(0): Config Properties File
    * @return result
    *         file with structure: M;k;finalPi,finalA,finalB
    */
  def main(args: Array[String]): Unit = {
    val applicationProps = new java.util.Properties()
    val in = new FileInputStream(args(0))
    applicationProps.load(in)
    in.close()

    if (applicationProps.getProperty("generate_logs").equals("true")) {
      Logger.getLogger("org").setLevel(Level.ERROR)
      Logger.getLogger("akka").setLevel(Level.ERROR)
    }

    val sparkSession = SparkSession.builder.appName("Spark-HMM").getOrCreate()

    val sample1: DataFrame = sparkSession.read.option("header", "true").csv(applicationProps.getProperty("path_sample1"))
      .sample(withReplacement = false, applicationProps.getProperty("size_sample").toDouble)
      .withColumnRenamed("_c0", "workitem").withColumnRenamed("_c1", "str_obs")

    val sample2: DataFrame = sparkSession.read.option("header", "true").csv(applicationProps.getProperty("path_sample2"))
      .sample(withReplacement = false, applicationProps.getProperty("size_sample").toDouble)
      .withColumnRenamed("_c0", "workitem").withColumnRenamed("_c1", "str_obs")

    val sample3: DataFrame = sparkSession.read.option("header", "true").csv(applicationProps.getProperty("path_sample3"))
      .sample(withReplacement = false, applicationProps.getProperty("size_sample").toDouble)
      .withColumnRenamed("_c0", "workitem").withColumnRenamed("_c1", "str_obs")

    val sample4: DataFrame = sparkSession.read.option("header", "true").csv(applicationProps.getProperty("path_sample4"))
      .sample(withReplacement = false, applicationProps.getProperty("size_sample").toDouble)
      .withColumnRenamed("_c0", "workitem").withColumnRenamed("_c1", "str_obs")

    val finalSample = sample1.union(sample2).union(sample3).union(sample4)
    val log = org.apache.log4j.LogManager.getRootLogger
    log.info("\n")
    log.info("Number of sequences: " + finalSample.count)

    // BaumWelchAlgorithm(observations, M, k, initialPi, initialA, initialB, numPartitions, epsilon, maxIterations)
    val result = hmm.BaumWelchAlgorithm.run1(finalSample,
      applicationProps.getProperty("value_M").toInt,
      applicationProps.getProperty("value_k").toInt,
      normalize(DenseVector.rand(applicationProps.getProperty("value_M").toInt), 1.0),
      hmm.Utils.mkstochastic(DenseMatrix.rand(applicationProps.getProperty("value_M").toInt, applicationProps.getProperty("value_M").toInt)),
      hmm.Utils.mkstochastic(DenseMatrix.rand(applicationProps.getProperty("value_M").toInt, applicationProps.getProperty("value_k").toInt)),
      applicationProps.getProperty("number_partitions").toInt,
      applicationProps.getProperty("value_epsilon").toDouble,
      applicationProps.getProperty("max_num_iterations").toInt)

    hmm.Utils.writeresult(applicationProps.getProperty("path_result"),
        applicationProps.getProperty("value_M") + ";" +
        applicationProps.getProperty("value_k") + ";" +
        result._1.toArray.mkString(",") + ";" +
        result._2.toArray.mkString(",") + ";" +
        result._3.toArray.mkString(","))
    sparkSession.stop()

  }
}
