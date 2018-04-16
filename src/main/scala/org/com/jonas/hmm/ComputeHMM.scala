package org.com.jonas.hmm

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.apache.spark.sql.expressions.MutableAggregationBuffer
import org.apache.spark.sql.expressions.UserDefinedAggregateFunction

class ComputeHMM extends UserDefinedAggregateFunction {
  // This is the input fields for your aggregate function.
  override def inputSchema: org.apache.spark.sql.types.StructType =
    StructType(StructField("M", IntegerType) ::
      StructField("k", IntegerType) ::
      StructField("T", IntegerType) ::
      StructField("gamma", StringType) ::
      StructField("current_ll", DoubleType) ::
      StructField("xi_summed", StringType) ::
      StructField("obs", StringType) :: Nil)

  // This is the internal fields you keep for computing your aggregate.
  override def bufferSchema: StructType = StructType(
    StructField("loglik", DoubleType) ::
      StructField("exp_num_trans", StringType) ::
      StructField("exp_num_visits1", StringType) ::
      StructField("exp_num_visitsT", StringType) ::
      StructField("exp_num_emit", StringType) :: Nil)

  // This is the output type of your aggregatation function.
  override def dataType: DataType = StringType

  override def deterministic: Boolean = true

  // This is the initial value for your buffer schema.
  override def initialize(buffer: MutableAggregationBuffer): Unit = {
    buffer(0) = 0.0
    buffer(1) = ""
    buffer(2) = ""
    buffer(3) = ""
    buffer(4) = ""
  }

  // This is how to update your buffer schema given an input.
  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    buffer(0) = buffer.getAs[Double](0) + input.getAs[Double](4)

    //Sumar matriz
    if (buffer.getAs[String](1) equals "") {
      buffer(1) = input.getAs[String](5)
    } else {
      buffer(1) = (buffer.getAs[String](1).split(",").map(_.toDouble),
        input.getAs[String](5).split(",").map(_.toDouble)).zipped.map(_ + _).mkString(",")
    }

    val gamma = new DenseMatrix(input.getAs[Int](0), input.getAs[Int](2), input.getAs[String](3).split(",").map(_.toDouble))
    if (buffer.getAs[String](2) equals "") {
      buffer(2) = gamma(::, 1).toArray.mkString(",")
    } else {
      buffer(2) = (buffer.getAs[String](2).split(",").map(_.toDouble),
        gamma(::, 1).toArray).zipped.map(_ + _).mkString(",")
    }

    if (buffer.getAs[String](3) equals "") {
      buffer(3) = gamma(::, input.getAs[Int](2) - 1).toArray.mkString(",")
    } else {
      buffer(3) = (buffer.getAs[String](3).split(",").map(_.toDouble),
        gamma(::, input.getAs[Int](2) - 1).toArray).zipped.map(_ + _).mkString(",")
    }

    var exp_num_emit = DenseMatrix.ones[Double](input.getAs[Int](0), input.getAs[Int](1))
    if (!(buffer.getAs[String](4) equals "")) {
      exp_num_emit = new DenseMatrix(input.getAs[Int](0), input.getAs[Int](1), buffer.getAs[String](4).split(",").map(_.toDouble))
    }
    val obs: Array[Int] = input.getAs[String](6).split(";").map(_.toInt)
    if (input.getAs[Int](2) < input.getAs[Int](1)) {
      (0 until input.getAs[Int](2)).foreach(t => {
        exp_num_emit(::, obs(t)) := exp_num_emit(::, obs(t)) + gamma(::, t)
      })
    } else {
      (0 until input.getAs[Int](1)).foreach(o => {
        val ndx = obs.zipWithIndex.filter(_._1 == o).map(_._2)
        if (ndx.length > 0) {
          val cont = DenseVector.zeros[Double](input.getAs[Int](0))
          ndx.foreach(i => cont := cont + gamma(::, i))
          exp_num_emit(::, o) := exp_num_emit(::, o) + cont
        }
      })
    }
    buffer(4) = exp_num_emit.toArray.mkString(",")
  }

  // This is how to merge two objects with the bufferSchema type.
  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    buffer1(0) = buffer1.getAs[Double](0) + buffer2.getAs[Double](0)

    if (buffer1.getAs[String](1) equals "") {
      buffer1(1) = buffer2.getAs[String](1)
    } else {
      buffer1(1) = (buffer1.getAs[String](1).split(",").map(_.toDouble),
        buffer2.getAs[String](1).split(",").map(_.toDouble)).zipped.map(_ + _).mkString(",")
    }

    if (buffer1.getAs[String](2) equals "") {
      buffer1(2) = buffer2.getAs[String](2)
    } else {
      buffer1(2) = (buffer1.getAs[String](2).split(",").map(_.toDouble),
        buffer2.getAs[String](2).split(",").map(_.toDouble)).zipped.map(_ + _).mkString(",")
    }

    if (buffer1.getAs[String](3) equals "") {
      buffer1(3) = buffer2.getAs[String](3)
    } else {
      buffer1(3) = (buffer1.getAs[String](3).split(",").map(_.toDouble),
        buffer2.getAs[String](3).split(",").map(_.toDouble)).zipped.map(_ + _).mkString(",")
    }

    if (buffer1.getAs[String](4) equals "") {
      buffer1(4) = buffer2.getAs[String](4)
    } else {
      buffer1(4) = (buffer1.getAs[String](4).split(",").map(_.toDouble),
        buffer2.getAs[String](4).split(",").map(_.toDouble)).zipped.map(_ + _).mkString(",")
    }
  }

  // This is where you output the final value, given the final value of your bufferSchema.
  override def evaluate(buffer: Row): Any = {
    buffer.getAs[Double](0).toString +
      ";" + buffer.getAs[String](1) +
      ";" + buffer.getAs[String](2) +
      ";" + buffer.getAs[String](3) +
      ";" + buffer.getAs[String](4)
  }
}
