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

    val sparkSession = SparkSession.builder.appName("Spark-HMM").getOrCreate()

    /**
      * Class 1 seq with error
      * Class 0 seq without error
      */
    val k_folds = applicationProps.getProperty("k_folds").toInt
    log.info("Value of k_folds: " + k_folds)
    val value_M = applicationProps.getProperty("value_M").toInt
    log.info("Value of value_M: " + value_M)
    val value_k = applicationProps.getProperty("value_k").toInt
    log.info("Value of value_k: " + value_k)
    val value_T = applicationProps.getProperty("value_T").toInt
    log.info("Value of value_T: " + value_T)
    val size_sample = applicationProps.getProperty("size_sample").toDouble
    log.info("Value of size_sample: " + size_sample)
    val number_partitions = applicationProps.getProperty("number_partitions").toInt
    log.info("Value of number_partitions: " + number_partitions)
    val value_epsilon = applicationProps.getProperty("value_epsilon").toDouble
    log.info("Value of value_epsilon: " + value_epsilon)
    val max_num_iterations = applicationProps.getProperty("max_num_iterations").toInt
    log.info("Value of max_num_iterations: " + max_num_iterations)

    var nClass1 = 0
    var nClass0 = 0
    var inInter = 0

    val path_sample_class1 = applicationProps.getProperty("sample_class1")
    log.info("Value of path_sample_class1: " + path_sample_class1)
    val path_sample_class0 = applicationProps.getProperty("sample_class0")
    log.info("Value of path_sample_class0: " + path_sample_class0)
    val path_sample_class1_folds = applicationProps.getProperty("sample_class1_folds")
    log.info("Value of path_sample_class1_folds: " + path_sample_class1_folds)
    val path_sample_class0_folds = applicationProps.getProperty("sample_class0_folds")
    log.info("Value of path_sample_class0_folds: " + path_sample_class0_folds)
    val path_result_models = applicationProps.getProperty("root_path") + applicationProps.getProperty("result_models")
    log.info("Value of path_result_models: " + path_result_models)
    val path_class1_models = applicationProps.getProperty("root_path") + applicationProps.getProperty("class1_models")
    log.info("Value of path_class1_models: " + path_class1_models)
    val path_class0_models = applicationProps.getProperty("root_path") + applicationProps.getProperty("class0_models")
    log.info("Value of path_class0_models: " + path_class0_models)
    val path_class1_iterations = applicationProps.getProperty("root_path") + applicationProps.getProperty("class1_iterations")
    log.info("Value of path_class1_iterations: " + path_class1_iterations)
    val path_class0_iterations = applicationProps.getProperty("root_path") + applicationProps.getProperty("class0_iterations")
    log.info("Value of path_class0_iterations: " + path_class0_iterations)
    val path_sample_class1_kfold = applicationProps.getProperty("root_path") + applicationProps.getProperty("sample_class1_kfold")
    log.info("Value of path_sample_class1_kfold: " + path_sample_class1_kfold)
    val path_sample_class0_kfold = applicationProps.getProperty("root_path") + applicationProps.getProperty("sample_class0_kfold")
    log.info("Value of path_sample_class0_kfold: " + path_sample_class0_kfold)
    val path_class1_models_baumwelch = applicationProps.getProperty("root_path") + applicationProps.getProperty("class1_models_baumwelch")
    log.info("Value of path_class1_models_baumwelch: " + path_class1_models_baumwelch)
    val path_class0_models_baumwelch = applicationProps.getProperty("root_path") + applicationProps.getProperty("class0_models_baumwelch")
    log.info("Value of path_class0_models_baumwelch: " + path_class0_models_baumwelch)

    if (new java.io.File(path_result_models).exists) {
      nClass1 = sparkSession.read.csv(path_sample_class1_folds).where("_c2 == 1").count().toInt
      log.info("Value of nClass1: " + nClass1)
      nClass0 = sparkSession.read.csv(path_sample_class0_folds).where("_c2 == 0").count().toInt
      log.info("Value of nClass0: " + nClass0)

      val file_result_models = scala.io.Source.fromFile(path_result_models)
      inInter = file_result_models.getLines.size - 1
      file_result_models.close()

    } else {
      /**
        * Make info Class 1
        */
      var sampleClass1 = sparkSession.read.csv(path_sample_class1)
        .where("_c2 == 1").drop("_c2")
        .sample(withReplacement = false, size_sample)
        .withColumnRenamed("_c0", "workitem")
        .withColumnRenamed("_c1", "str_obs")
        .select(col("workitem"), col("str_obs"),
          row_number().over(Window.orderBy(col("workitem"))).alias("rowId"))
      nClass1 = sampleClass1.count().toInt
      log.info("Value of nClass1: " + nClass1)

      sampleClass1 = set_folds(sampleClass1, nClass1, k_folds)
      sampleClass1.write.format("com.databricks.spark.csv").save(path_sample_class1_folds)
      sampleClass1.unpersist()

      /**
        * Make info Class 0
        */
      var sampleClass0 = sparkSession.read.csv(path_sample_class0)
        .where("_c2 == 0").drop("_c2")
        .sample(withReplacement = false, size_sample)
        .withColumnRenamed("_c0", "workitem")
        .withColumnRenamed("_c1", "str_obs")
        .select(col("workitem"), col("str_obs"),
          row_number().over(Window.orderBy(col("workitem"))).alias("rowId"))
      nClass0 = sampleClass0.count().toInt
      log.info("Value of nClass0: " + nClass0)

      sampleClass0 = set_folds(sampleClass0, nClass0, k_folds)
      sampleClass0.write.format("com.databricks.spark.csv").save(path_sample_class0_folds)
      sampleClass0.unpersist()

      hmm.Utils.writeresult(path_result_models,"N,TP,FP,FN,TN,sensitivity,specificity,accuracy,error,g-mean\n")
      hmm.Utils.writeresult(path_class1_models,"kfold;M;k;Pi;A;B\n")
      hmm.Utils.writeresult(path_class0_models,"kfold;M;k;Pi;A;B\n")
    }

    (inInter until k_folds).foreach(inter => {
      log.info("*****************************************************************************************")
      log.info("Fold number: " + inter)
      log.info("*****************************************************************************************")

      var modelClass1 = (Array.empty[Double], Array.empty[Double], Array.empty[Double])
      var modelClass0 = (Array.empty[Double], Array.empty[Double], Array.empty[Double])

      val file_class1_models = scala.io.Source.fromFile(path_class1_models)
      val file_class1_size = file_class1_models.getLines.size
      val class1_string_model: List[String] = file_class1_models.getLines().toList
      file_class1_models.close()

      if (file_class1_size == inter + 2) {
        log.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        log.info("Start Load Model Class 1")
        val arraymodel = class1_string_model.last.split(";")
        modelClass1 = (
          arraymodel(3).split(",").map(_.toDouble),
          arraymodel(4).split(",").map(_.toDouble),
          arraymodel(5).split(",").map(_.toDouble))
        log.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

      } else {
        log.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        hmm.Utils.writeresult(path_class1_iterations,"Fold number: " + inter + "\n")
        hmm.Utils.writeresult(path_class1_iterations,"iteration;minutes;hours;loglik;antloglik;epsilon\n")

        log.info("Getting data to train Class 1")
        val trainClass1 = sparkSession.read.csv(path_sample_class1_folds)
          .where("_c2 == 1").drop("_c2")
          .withColumnRenamed("_c0", "workitem")
          .withColumnRenamed("_c1", "str_obs")
          .withColumnRenamed("_c3", "kfold")
          .where("kfold <> " + inter).drop("kfold")
        trainClass1.persist()

        log.info("Start training Class 1")
        val tmpModelClass1 = hmm.BaumWelchAlgorithm.run(trainClass1, value_M, value_k, value_T,
          normalize(DenseVector.rand(value_M).asInstanceOf[DenseVector[Double]], 1.0),
          hmm.Utils.mkstochastic(DenseMatrix.rand(value_M, value_M)),
          hmm.Utils.mkstochastic(DenseMatrix.rand(value_M, value_k)),
          number_partitions, value_epsilon, max_num_iterations,
          inter, path_class1_models_baumwelch, path_class1_iterations)

        modelClass1 = (tmpModelClass1._1.toArray, tmpModelClass1._2.toArray, tmpModelClass1._3.toArray)
        hmm.Utils.writeresult(path_class1_models, inter + ";" + value_M + ";" + value_k + ";" +
            modelClass1._1.mkString(",") + ";" +
            modelClass1._2.mkString(",") + ";" +
            modelClass1._3.mkString(",") + "\n")
        trainClass1.unpersist()
        log.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      }

      val file_class0_models = scala.io.Source.fromFile(path_class0_models)
      val file_class0_size = file_class0_models.getLines.size
      val class0_string_model: List[String] = file_class0_models.getLines().toList
      file_class0_models.close()

      if (file_class0_size == inter + 2) {
        log.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        log.info("Start Load Model Class 0")
        val arraymodel = class0_string_model.last.split(";")
        modelClass0 = (
          arraymodel(3).split(",").map(_.toDouble),
          arraymodel(4).split(",").map(_.toDouble),
          arraymodel(5).split(",").map(_.toDouble))
        log.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

      } else {
        log.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        hmm.Utils.writeresult(path_class0_iterations,"Fold number: " + inter + "\n")
        hmm.Utils.writeresult(path_class0_iterations, "iteration;minutes;hours;loglik;antloglik;epsilon\n")

        log.info("Getting data to train Class 0")
        val trainClass0 = sparkSession.read.csv(path_sample_class0_folds)
          .where("_c2 == 0").drop("_c2")
          .withColumnRenamed("_c0", "workitem")
          .withColumnRenamed("_c1", "str_obs")
          .withColumnRenamed("_c3", "kfold")
          .where("kfold <> " + inter).drop("kfold")
        trainClass0.persist()

        log.info("Start training Class 0")
        val tmpModelClass0 = hmm.BaumWelchAlgorithm.run(trainClass0, value_M, value_k, value_T,
          normalize(DenseVector.rand(value_M).asInstanceOf[DenseVector[Double]], 1.0),
          hmm.Utils.mkstochastic(DenseMatrix.rand(value_M, value_M)),
          hmm.Utils.mkstochastic(DenseMatrix.rand(value_M, value_k)),
          number_partitions, value_epsilon, max_num_iterations,
          inter, path_class0_models_baumwelch, path_class0_iterations)

        modelClass0 = (tmpModelClass0._1.toArray, tmpModelClass0._2.toArray, tmpModelClass0._3.toArray)
        hmm.Utils.writeresult(path_class0_models, inter + ";" + value_M + ";" + value_k + ";" +
            modelClass0._1.mkString(",") + ";" +
            modelClass0._2.mkString(",") + ";" +
            modelClass0._3.mkString(",") + "\n")
        trainClass0.unpersist()
        log.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      }

      log.info("Getting data to validate Class 1")
      val validClass1 = sparkSession.read.csv(path_sample_class1_folds)
        .where("_c2 == 1").drop("_c2")
        .withColumnRenamed("_c0", "workitem")
        .withColumnRenamed("_c1", "str_obs")
        .withColumnRenamed("_c3", "kfold")
        .where("kfold == " + inter).drop("kfold")
      validClass1.persist()

      val resultClass1 =
        hmm.BaumWelchAlgorithm.validate(validClass1, value_M, value_k, value_T, number_partitions,
          new DenseVector(modelClass1._1), new DenseMatrix(value_M, value_M, modelClass1._2), new DenseMatrix(value_M, value_k, modelClass1._3))
          .withColumnRenamed("prob", "probMod1")
          .join(
            hmm.BaumWelchAlgorithm.validate(validClass1, value_M, value_k, value_T, number_partitions,
              new DenseVector(modelClass0._1), new DenseMatrix(value_M, value_M, modelClass0._2), new DenseMatrix(value_M, value_k, modelClass0._3))
              .withColumnRenamed("prob", "probMod0"),
            Seq("workitem"), "inner")
          .select(col("workitem"), col("probMod1"), col("probMod0"))
      resultClass1.persist()

      log.info("Saving result validation Class1")
      log.info("Value of validClass1: " + validClass1.count())
      validClass1.unpersist()
      resultClass1.write.format("com.databricks.spark.csv").save(path_sample_class1_kfold + inter)
      log.info("Value of resultClass1: " + resultClass1.count())

      log.info("Getting data to validate Class 0")
      val validClass0 = sparkSession.read.csv(path_sample_class0_folds)
        .where("_c2 == 0").drop("_c2")
        .withColumnRenamed("_c0", "workitem")
        .withColumnRenamed("_c1", "str_obs")
        .withColumnRenamed("_c3", "kfold")
        .where("kfold == " + inter).drop("kfold")
      validClass0.persist()

      val resultClass0 =
        hmm.BaumWelchAlgorithm.validate(validClass0, value_M, value_k, value_T, number_partitions,
          new DenseVector(modelClass1._1), new DenseMatrix(value_M, value_M, modelClass1._2), new DenseMatrix(value_M, value_k, modelClass1._3))
          .withColumnRenamed("prob", "probMod1")
          .join(
            hmm.BaumWelchAlgorithm.validate(validClass0, value_M, value_k, value_T, number_partitions,
              new DenseVector(modelClass0._1), new DenseMatrix(value_M, value_M, modelClass0._2), new DenseMatrix(value_M, value_k, modelClass0._3))
              .withColumnRenamed("prob", "probMod0"),
            Seq("workitem"), "inner")
          .select(col("workitem"), col("probMod1"), col("probMod0"))

      log.info("Saving result validation Class0")
      log.info("Value of validClass0: " + validClass0.count())
      validClass0.unpersist()
      resultClass0.write.format("com.databricks.spark.csv").save(path_sample_class0_kfold + inter)
      log.info("Value of resultClass0: " + resultClass0.count())

      /** N value */
      log.info("Compute N")
      val N: Double = resultClass1.count + resultClass0.count
      log.info("Value of N: " + N)

      /** True Positives */
      log.info("Compute True Positives")
      val TP: Double = resultClass1.where("probMod1 > probMod0").count
      log.info("Value of TP: " + TP)

      /** False Positives */
      log.info("Compute False Positives")
      val FP: Double = resultClass0.where("probMod1 > probMod0").count
      log.info("Value of FP: " + FP)

      /** False Negatives */
      log.info("Compute False Negatives")
      val FN: Double = resultClass1.where("probMod1 < probMod0").count
      log.info("Value of FN: " + FN)

      /** True Negatives */
      log.info("Compute True Negatives")
      val TN: Double = resultClass0.where("probMod1 < probMod0").count
      log.info("Value of TN: " + TN)

      /** sensitivity */
      log.info("Compute Sensitivity")
      val sensi: Double = TP / (TP + FN)
      log.info("Value of sensi: " + sensi)

      /** specificity */
      log.info("Compute Specificity")
      val speci: Double = TN / (TN + FP)
      log.info("Value of speci: " + speci)

      /** accuracy */
      log.info("Compute Accuracy")
      val effic: Double = (TP + TN) / (TP + FP + FN + TN)
      log.info("Value of accuracy: " + effic)

      /** error */
      log.info("Compute Error")
      val error: Double = 1 - effic
      log.info("Value of error: " + error)

      /** g-mean */
      log.info("Compute G-mean")
      val gmean = scala.math.sqrt(sensi * speci)
      log.info("Value of g-mean: " + gmean)

      log.info("*****************************************************************************************")
      hmm.Utils.writeresult(path_result_models, N + "," + TP + "," + FP + "," + FN + "," + TN + "," +
        sensi + "," + speci + "," + effic + "," + error + "," + gmean + "\n")

      resultClass1.unpersist()
      resultClass0.unpersist()
    })
    sparkSession.stop()
  }

  def set_folds(sample: DataFrame, n: Int, kfolds: Int): DataFrame = {
    val randomList = scala.util.Random.shuffle((0 until n).toList)
    val indexList = Array.fill[Int](n)(0)
    (1 until kfolds).foreach(i => (i until n by kfolds).foreach(j => indexList(randomList(j)) = i))
    val udf_setfold: UserDefinedFunction = udf((rowId: Int) => indexList(rowId - 1))
    sample.withColumn("kfold", udf_setfold(col("rowId")))
  }
}
