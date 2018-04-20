package org.com.jonas

import java.io.FileInputStream
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession}
import breeze.linalg.{DenseMatrix, DenseVector, normalize}

object Train {
  /**
    * @param args
    * args(0): Path of properties
    *
    * @return result
    * file with structure: M;k;finalPi,finalA,finalB
    */
  def main(args: Array[String]): Unit = {
    val applicationProps = new java.util.Properties()
    val in = new FileInputStream("args(0)")
    applicationProps.load(in)
    in.close()

    if(applicationProps.getProperty("generate_logs").equals("true")){
      Logger.getLogger("org").setLevel(Level.ERROR)
      Logger.getLogger("akka").setLevel(Level.ERROR)
    }

    val sparkSession = SparkSession.builder.appName("Spark-HMM").getOrCreate()
    val sample: DataFrame = sparkSession.read.csv(applicationProps.getProperty("path_sample"))
      .sample(withReplacement = false, applicationProps.getProperty("size_sample").toDouble)
      .withColumnRenamed("_c0", "workitem").withColumnRenamed("_c1", "str_obs")

    // BaumWelchAlgorithm(M, k, initialPi, initialA, initialB, numPartitions, epsilon, maxIterations)
    val result = hmm.BaumWelchAlgorithm.run(sample,
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
