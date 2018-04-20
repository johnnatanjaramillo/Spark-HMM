package org.com.jonas

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession}
import breeze.linalg.{DenseMatrix, DenseVector, normalize}

object Train {
  /**
    * @param args
    * args(0): Generate logs
    * args(1): Path of sample
    * args(2): Size of sample
    * args(3): Value of M
    * args(4): Value of k
    * args(5): Number of partitions (fw-bw process)
    * args(6): Value of epsilon
    * args(7): Max num iterations
    * args(8): Path to result
    *
    * @return result
    * file with structure: M;k;finalPi,finalA,finalB
    */
  def main(args: Array[String]): Unit = {
    if(args(0).equals("1")){
      Logger.getLogger("org").setLevel(Level.ERROR)
      Logger.getLogger("akka").setLevel(Level.ERROR)
    }
    val sparkSession = SparkSession.builder.appName("Spark-HMM").getOrCreate()
    val sample: DataFrame = sparkSession.read.csv(args(1)).sample(withReplacement = false, args(2).toDouble)
      .withColumnRenamed("_c0", "workitem").withColumnRenamed("_c1", "str_obs")
    // BaumWelchAlgorithm(M, k, initialPi, initialA, initialB, numPartitions)
    val result = hmm.BaumWelchAlgorithm.run(sample, args(3).toInt, args(4).toInt,
      normalize(DenseVector.rand(args(3).toInt), 1.0),
      hmm.Utils.mkstochastic(DenseMatrix.rand(args(3).toInt, args(3).toInt)),
      hmm.Utils.mkstochastic(DenseMatrix.rand(args(3).toInt, args(4).toInt)),
      args(5).toInt, args(6).toDouble, args(7).toInt)
    hmm.Utils.writeresult(args(8),
      args(3) + ";" +
        args(4) + ";" +
        result._1.toArray.mkString(",") + ";" +
        result._2.toArray.mkString(",") + ";" +
        result._3.toArray.mkString(","))
    sparkSession.stop()
  }
}
