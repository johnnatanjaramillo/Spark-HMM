package org.com.jonas

import java.io.FileInputStream

import breeze.linalg.{DenseMatrix, DenseVector, normalize, sum}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{col, row_number}
import org.com.jonas.CrossValidation.set_folds

object testProb {

  def main(args: Array[String]): Unit = {


    val stringModel_0: List[String] = scala.io.Source.fromFile("C:\\Users\\johnnatan.jaramillo\\Downloads\\Modelos\\ModelsClass1KmeansM10.csv").getLines().toList
    val stringModel_1: List[String] = scala.io.Source.fromFile("C:\\Users\\johnnatan.jaramillo\\Downloads\\Modelos\\ModelsClass1KmeansM10.csv").getLines().toList

    val M = stringModel_0(1).split(";")(1).toInt
    val k = stringModel_0(1).split(";")(2).toInt


    val funPi: DenseVector[Double] = new DenseVector(stringModel_0(1).split(";")(3).split(",").map(_.toDouble))
    val funA = new DenseMatrix(M, M, stringModel_0(1).split(";")(4).split(",").map(_.toDouble))

    val funB: DenseMatrix[Double] = new DenseMatrix(M, k, stringModel_0(1).split(";")(5).split(",").map(_.toDouble))

    val seq: List[String] = scala.io.Source.fromFile("C:\\Users\\johnnatan.jaramillo\\Downloads\\Modelos\\seqkmeans2\\part-00001-75357c56-cb6f-4bf4-ae5f-09ab6e1ead16.csv").getLines().toList

    val T = seq(0).split(",")(1).split(";").size
    val obs = seq(0).split(",")(1).split(";").map(_.toInt)

    val output: DenseMatrix[Double] = DenseMatrix.tabulate(M, T) { case (m, t) => funB(m, obs(t)) }
    val funObslik = new DenseMatrix(M, T, output.toArray)

    val alpha: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, T)
    alpha(::, 0) := normalize(funPi :* funObslik(::, 0), 1.0)
    println(sum(alpha(::, 0)))
    (1 until T).foreach(t => {
      alpha(::, t) := normalize((funA.t * alpha(::, t - 1)) :* funObslik(::, t), 1.0)
      println(sum(alpha(::, t)))
    })
    println(sum(alpha(::, T - 1)))


  }

}
