package org.com.jonas

import java.io.FileInputStream

import breeze.linalg.{DenseMatrix, DenseVector, normalize}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.{UserDefinedFunction, Window}

object CrossValidation {
  /**
    * @param args
    * args(0): Config Properties File
    * @return result
    *
    */
  def main(args: Array[String]): Unit = {
    val log = org.apache.log4j.LogManager.getRootLogger
    val applicationProps = new java.util.Properties()
    val in = new FileInputStream(args(0))
    applicationProps.load(in)
    in.close()

    if (applicationProps.getProperty("generate_logs").equals("true")) {
      Logger.getLogger("org").setLevel(Level.ERROR)
      Logger.getLogger("akka").setLevel(Level.ERROR)
    }

    val k_folds = applicationProps.getProperty("k_folds").toInt
    log.info("Value of k_folds: " + k_folds)
    val value_M = applicationProps.getProperty("value_M").toInt
    log.info("Value of value_M: " + value_M)
    val value_k = applicationProps.getProperty("value_k").toInt
    log.info("Value of value_k: " + value_k)
    val number_partitions = applicationProps.getProperty("number_partitions").toInt
    log.info("Value of number_partitions: " + number_partitions)
    val value_epsilon = applicationProps.getProperty("value_epsilon").toDouble
    log.info("Value of value_epsilon: " + value_epsilon)
    val max_num_iterations = applicationProps.getProperty("max_num_iterations").toInt
    log.info("Value of max_num_iterations: " + max_num_iterations)

    /**
      * Class 1 seq with error
      * Class 0 seq without error
      */
    val sparkSession = SparkSession.builder.appName("Spark-HMM").getOrCreate()
    /**
      * Make info Class 1
      */
    var sampleClass1 = sparkSession.read.option("header", "true").csv(applicationProps.getProperty("path_sample_Class1"))
      .sample(withReplacement = false, applicationProps.getProperty("size_sample").toDouble)
      .withColumnRenamed("_c0", "workitem").withColumnRenamed("_c1", "str_obs")
      .select(col("workitem"), col("str_obs"), row_number().over(Window.orderBy(col("workitem"))).alias("rowId"))

    val nClass1 = sampleClass1.count().toInt
    log.info("Value of nClass1: " + nClass1)
    sampleClass1 = set_folds(sampleClass1, nClass1, k_folds)

    /**
      * Make info Class 0
      */
    var sampleClass0 = sparkSession.read.option("header", "true").csv(applicationProps.getProperty("path_sample_Class0"))
      .sample(withReplacement = false, applicationProps.getProperty("size_sample").toDouble)
      .withColumnRenamed("_c0", "workitem").withColumnRenamed("_c1", "str_obs")
      .select(col("workitem"), col("str_obs"), row_number().over(Window.orderBy(col("workitem"))).alias("rowId"))

    val nClass0 = sampleClass0.count().toInt
    log.info("Value of nClass0: " + nClass0)
    sampleClass0 = set_folds(sampleClass0, nClass0, k_folds)

    //**//
    //(0 until kfolds).foreach(iter => {
    log.info("Getting data for train class 1")
    val trainClass1 = sampleClass1.where("kfold <> " + 0).drop(col("kfold")).drop(col("rowId"))
    log.info("Getting data for train class 0")
    val trainClass0 = sampleClass0.where("kfold <> " + 0).drop(col("kfold")).drop(col("rowId"))
    log.info("Getting data for validate class 1")
    val validClass1 = sampleClass1.where("kfold == " + 0).drop(col("kfold")).drop(col("rowId"))
    log.info("Getting data for validate class 0")
    val validClass0 = sampleClass0.where("kfold == " + 0).drop(col("kfold")).drop(col("rowId"))

    log.info("Start training class 1")
    val modelClass1 = hmm.BaumWelchAlgorithm.run1(trainClass1, value_M, value_k,
      normalize(DenseVector.rand(value_M), 1.0),
      hmm.Utils.mkstochastic(DenseMatrix.rand(value_M, value_k)),
      hmm.Utils.mkstochastic(DenseMatrix.rand(value_M, value_k)),
      number_partitions, value_epsilon, max_num_iterations)

    log.info("Start training class 0")
    val modelClass0 = hmm.BaumWelchAlgorithm.run1(trainClass0, value_M, value_k,
      normalize(DenseVector.rand(value_M), 1.0),
      hmm.Utils.mkstochastic(DenseMatrix.rand(value_M, value_k)),
      hmm.Utils.mkstochastic(DenseMatrix.rand(value_M, value_k)),
      number_partitions, value_epsilon, max_num_iterations)

    /** True Positives */
    log.info("Compute True Positives")
    val TP = hmm.BaumWelchAlgorithm.validate(validClass1, value_M, value_k, modelClass1._1, modelClass1._2, modelClass1._3)
        .where("prob > 0.5").count
    log.info("Value of TP " + TP)

    /** False Positives */
    log.info("Compute False Positives")
    val FP = hmm.BaumWelchAlgorithm.validate(validClass0, value_M, value_k, modelClass1._1, modelClass1._2, modelClass1._3)
      .where("prob > 0.5").count
    log.info("Value of FP " + FP)

    /** False Negatives */
    log.info("Compute False Negatives")
    val FN = hmm.BaumWelchAlgorithm.validate(validClass1, value_M, value_k, modelClass0._1, modelClass0._2, modelClass0._3)
      .where("prob > 0.5").count
    log.info("Value of FN " + FN)

    /** True Negatives */
    log.info("Compute True Negatives")
    val VN = hmm.BaumWelchAlgorithm.validate(validClass0, value_M, value_k, modelClass0._1, modelClass0._2, modelClass0._3)
      .where("prob > 0.5").count
    log.info("Value of VN " + VN)

    //})
    sparkSession.stop()
  }

  def set_folds(sample: DataFrame, n: Int, kfolds: Int): DataFrame ={
    val randomList = scala.util.Random.shuffle((0 until n).toList)
    val indexList = Array.fill[Int](n)(0)
    (1 until kfolds).foreach(i => (i until n by kfolds).foreach(j => indexList(randomList(j)) = i))

    val udf_setfold: UserDefinedFunction = udf((rowId: Int) => indexList(rowId - 1))
    sample.withColumn("kfold", udf_setfold(col("rowId")))
  }

}
