package org.com.jonas

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession}
import breeze.linalg.{DenseMatrix, DenseVector, normalize}

object Train {
  /**
    * @param args
    * args(0): Path of sample
    * args(1): Size of sample
    * args(2): Value of M
    * args(3): Value of k
    * args(4): Number of partitions (fw-bw process)
    * args(5): Path to result
    *
    * @return result
    * file with structure: M;k;finalPi,finalA,finalB
    */
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    val sparkSession = SparkSession.builder.appName("Spark-HMM").getOrCreate()
    val sample: DataFrame = sparkSession.read.csv(args(0)).sample(withReplacement = false, args(1).toDouble)
      .withColumnRenamed("_c0", "workitem").withColumnRenamed("_c1", "str_obs")
    // BaumWelchAlgorithm(M, k, initialPi, initialA, initialB, numPartitions)
    val result = hmm.BaumWelchAlgorithm.run(sample, args(2).toInt, args(3).toInt,
      normalize(DenseVector.rand(args(2).toInt), 1.0),
      hmm.Utils.mkstochastic(DenseMatrix.rand(args(2).toInt, args(2).toInt)),
      hmm.Utils.mkstochastic(DenseMatrix.rand(args(2).toInt, args(3).toInt)),
      args(4).toInt)
    hmm.Utils.writeresult(args(5),
      args(2) + ";" +
        args(3) + ";" +
        result._1.toArray.mkString(",") + ";" +
        result._2.toArray.mkString(",") + ";" +
        result._3.toArray.mkString(","))
    sparkSession.stop()
  }
}
