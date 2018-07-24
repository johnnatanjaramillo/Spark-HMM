package org.com.jonas

import java.io.FileInputStream

import breeze.linalg.{DenseMatrix, DenseVector, max, normalize, sum}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{col, row_number}
import org.com.jonas.CrossValidation.set_folds
import org.com.jonas.hmm.Utils

object testProb {

  def main(args: Array[String]): Unit = {


    val stringModel_0: List[String] = scala.io.Source.fromFile("C:\\Users\\johnnatan.jaramillo\\Downloads\\Modelos\\ModelsClass0KmeansM10.csv").getLines().toList
    val stringModel_1: List[String] = scala.io.Source.fromFile("C:\\Users\\johnnatan.jaramillo\\Downloads\\Modelos\\ModelsClass1KmeansM10.csv").getLines().toList

    val M = stringModel_0(1).split(";")(1).toInt
    val k = stringModel_0(1).split(";")(2).toInt


    val funPi: DenseVector[Double] = new DenseVector(stringModel_0(1).split(";")(3).split(",").map(_.toDouble))
    val funA = new DenseMatrix(M, M, stringModel_0(1).split(";")(4).split(",").map(_.toDouble))

    val funB: DenseMatrix[Double] = new DenseMatrix(M, k, stringModel_0(1).split(";")(5).split(",").map(_.toDouble))

    val seq: List[String] = scala.io.Source.fromFile("C:\\Users\\johnnatan.jaramillo\\Downloads\\Modelos\\seqkmeans\\part-00003-c9f97853-2fa9-4ec1-89a0-16d53a680cf7.csv").getLines().toList

    val T = seq(0).split(",")(1).split(";").size
    val obs = seq(0).split(",")(1).split(";").map(_.toInt)

    val output: DenseMatrix[Double] = DenseMatrix.tabulate(M, T) { case (m, t) => funB(m, obs(t)) }
    val funObslik = new DenseMatrix(M, T, output.toArray)

    val scale: DenseVector[Double] = DenseVector.ones[Double](T)
    val alpha: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, T)

    alpha(::, 0) := Utils.normalise(funPi :* funObslik(::, 0), scale, 0)
    (1 until T).foreach(t => alpha(::, t) := Utils.normalise((funA.t * alpha(::, t - 1)) :* funObslik(::, t), scale, t))

    val loglik: Double = sum(scale.map(Math.log))

    println(loglik)

  }

}
