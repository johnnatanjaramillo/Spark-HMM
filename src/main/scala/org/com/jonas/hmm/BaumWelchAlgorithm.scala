package org.com.jonas.hmm

import scala.util.control.Breaks._
import breeze.linalg.{DenseMatrix, DenseVector, normalize, sum}
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.UserDefinedFunction

object BaumWelchAlgorithm {
  def run(observations: DataFrame, M: Int, k: Int,
          initialPi: DenseVector[Double], initialA: DenseMatrix[Double], initialB: DenseMatrix[Double],
          numPartitions: Int = 1, epsilon: Double = 0.0001, maxIterations: Int = 10000):
  (DenseVector[Double], DenseMatrix[Double], DenseMatrix[Double]) = {

    var prior = initialPi
    var transmat = initialA
    var obsmat = initialB
    var antloglik: Double = Double.NegativeInfinity

    observations.persist()
    var obstrained = observations
      .withColumn("M", lit(M))
      .withColumn("k", lit(k))
      .withColumn("Pi", lit(initialPi.toArray))
      .withColumn("A", lit(initialA.toArray))
      .withColumn("B", lit(initialB.toArray))
      .withColumn("obs", udf_toarray(col("str_obs")))
      .withColumn("T", udf_obssize(col("obs")))

    breakable {
      (0 until maxIterations).foreach(_ => {

        val computehmm = new ComputeHMM
        val newvalues = obstrained.repartition(numPartitions)
          .withColumn("obslik", udf_multinomialprob(col("obs"), col("M"), col("k"), col("T"), col("B")))
          .withColumn("fwdback", udf_fwdback(col("M"), col("T"), col("Pi"), col("A"), col("obslik")))
          .withColumn("gamma", udf_gamma_str(col("fwdback")))
          .withColumn("loglik", udf_loglik(col("fwdback")))
          .withColumn("xi_summed", udf_xi_summed_str(col("fwdback")))
          .drop("workitem", "Pi", "A", "B", "obs", "obslik", "fwdback")
          .agg(computehmm(col("M"), col("k"), col("T"), col("gamma"), col("loglik"), col("xi_summed"), col("str_obs")).as("ess"))
          .head().getAs[String]("ess").split(";")

        val loglik = newvalues(0).toDouble
        prior = normalize(new DenseVector(newvalues(2).split(",").map(_.toDouble)), 1.0)
        transmat = Utils.mkstochastic(new DenseMatrix(M, M, newvalues(1).split(",").map(_.toDouble)))
        obsmat = Utils.mkstochastic(new DenseMatrix(M, k, newvalues(4).split(",").map(_.toDouble)))

        if (Utils.emconverged(loglik, antloglik, epsilon)) break
        antloglik = loglik

        obstrained.unpersist()
        obstrained = observations
          .withColumn("M", lit(M))
          .withColumn("k", lit(k))
          .withColumn("Pi", lit(prior.toArray))
          .withColumn("A", lit(transmat.toArray))
          .withColumn("B", lit(obsmat.toArray))
          .withColumn("obs", udf_toarray(col("str_obs")))
          .withColumn("T", udf_obssize(col("obs")))

      })
    }
    (prior, transmat, obsmat)
  }

  def run1(observations: DataFrame, M: Int, k: Int,
           initialPi: DenseVector[Double], initialA: DenseMatrix[Double], initialB: DenseMatrix[Double],
           numPartitions: Int = 1, epsilon: Double = 0.0001, maxIterations: Int = 10000):
  (DenseVector[Double], DenseMatrix[Double], DenseMatrix[Double]) = {

    var prior = initialPi
    var transmat = initialA
    var obsmat = initialB
    var antloglik: Double = Double.NegativeInfinity
    val log = org.apache.log4j.LogManager.getRootLogger

    observations.persist()
    var obstrained = observations
      .withColumn("M", lit(M))
      .withColumn("k", lit(k))
      .withColumn("Pi", lit(initialPi.toArray))
      .withColumn("A", lit(initialA.toArray))
      .withColumn("B", lit(initialB.toArray))
      .withColumn("obs", udf_toarray(col("str_obs")))
      .withColumn("T", udf_obssize(col("obs")))

    breakable {
      (0 until maxIterations).foreach(it => {
        log.info("\n")
        log.info("Start Iteration: " + it)
        val computehmm = new ComputeHMM
        val newvalues = obstrained.repartition(numPartitions)
          .withColumn("obslik", udf_multinomialprob(col("obs"), col("M"), col("k"), col("T"), col("B")))
          .withColumn("fwdback", udf_fwdback(col("M"), col("T"), col("Pi"), col("A"), col("obslik")))
          .withColumn("gamma", udf_gamma(col("fwdback")))
          .withColumn("loglik", udf_loglik(col("fwdback")))
          .withColumn("xi_summed", udf_xi_summed(col("fwdback")))
          .withColumn("exp_num_visits1", udf_exp_num_visits1(col("gamma"), col("M"), col("T")))
          .withColumn("exp_num_emit", udf_exp_num_emit(col("gamma"), col("M"), col("k"), col("T"), col("obs")))
          .drop("workitem", "str_obs", "M", "k", "Pi", "A", "B", "obs", "T", "obslik", "fwdback", "gamma")
          .rdd.treeReduce((row1, row2) =>
          Row(row1.getAs[Double](0) + row2.getAs[Double](0),
            (row1.getAs[Seq[Double]](1), row2.getAs[Seq[Double]](1)).zipped.map(_ + _),
            (row1.getAs[Seq[Double]](2), row2.getAs[Seq[Double]](2)).zipped.map(_ + _),
            (row1.getAs[Seq[Double]](3), row2.getAs[Seq[Double]](3)).zipped.map(_ + _)))

        val loglik = newvalues.getAs[Double](0)
        log.info("\n")
        log.info("LogLikehood Value: " + loglik)
        prior = normalize(new DenseVector(newvalues.getAs[Seq[Double]](2).toArray), 1.0)
        transmat = Utils.mkstochastic(new DenseMatrix(M, M, newvalues.getAs[Seq[Double]](1).toArray))
        obsmat = Utils.mkstochastic(new DenseMatrix(M, k, newvalues.getAs[Seq[Double]](3).toArray))

        if (Utils.emconverged(loglik, antloglik, epsilon)) break
        antloglik = loglik

        obstrained.unpersist()
        obstrained = observations
          .withColumn("M", lit(M))
          .withColumn("k", lit(k))
          .withColumn("Pi", lit(prior.toArray))
          .withColumn("A", lit(transmat.toArray))
          .withColumn("B", lit(obsmat.toArray))
          .withColumn("obs", udf_toarray(col("str_obs")))
          .withColumn("T", udf_obssize(col("obs")))
        log.info("\n")
        log.info("End Iteration: " + it)
      })
    }
    (prior, transmat, obsmat)
  }

  def run2(observations: DataFrame, M: Int, k: Int,
           initialPi: DenseVector[Double], initialA: DenseMatrix[Double], initialB: DenseMatrix[Double],
           numPartitions: Int = 1, epsilon: Double = 0.0001, maxIterations: Int = 10000):
  (DenseVector[Double], DenseMatrix[Double], DenseMatrix[Double]) = {

    var prior = initialPi
    var transmat = initialA
    var obsmat = initialB
    var antloglik: Double = Double.NegativeInfinity
    val log = org.apache.log4j.LogManager.getRootLogger

    observations.persist()
    var obstrained = observations
      .withColumn("M", lit(M))
      .withColumn("k", lit(k))
      .withColumn("Pi", lit(initialPi.toArray))
      .withColumn("A", lit(initialA.toArray))
      .withColumn("B", lit(initialB.toArray))
      .withColumn("obs", udf_toarray(col("str_obs")))
      .withColumn("T", udf_obssize(col("obs")))

    breakable {
      (0 until maxIterations).foreach(it => {
        log.info("\n")
        log.info("Start Iteration: " + it)
        val computehmm = new ComputeHMM
        val newvalues = obstrained.repartition(numPartitions)
          .withColumn("obslik", udf_multinomialprob(col("obs"), col("M"), col("k"), col("T"), col("B")))
//          .withColumn("fwdback", udf_fwdback(col("M"), col("T"), col("Pi"), col("A"), col("obslik")))
//          .withColumn("gamma", udf_gamma(col("fwdback")))
//          .withColumn("loglik", udf_loglik(col("fwdback")))
//          .withColumn("xi_summed", udf_xi_summed(col("fwdback")))
//          .withColumn("exp_num_visits1", udf_exp_num_visits1(col("gamma"), col("M"), col("T")))
//          .withColumn("exp_num_emit", udf_exp_num_emit(col("gamma"), col("M"), col("k"), col("T"), col("obs")))
          .drop("workitem", "str_obs", "M", "k", "Pi", "A", "B", "obs", "T", "obslik", "fwdback", "gamma")
          .reduce((r1, r2) => r1)

        log.info("\n")
        log.info("End Iteration: " + it)
      })
    }
    (prior, transmat, obsmat)
  }

  def validate(observations: DataFrame, M: Int, k: Int,
               initialPi: DenseVector[Double], initialA: DenseMatrix[Double], initialB: DenseMatrix[Double]):
  DataFrame = {
    observations
      .withColumn("M", lit(M))
      .withColumn("k", lit(k))
      .withColumn("Pi", lit(initialPi.toArray))
      .withColumn("A", lit(initialA.toArray))
      .withColumn("B", lit(initialB.toArray))
      .withColumn("obs", udf_toarray(col("str_obs")))
      .withColumn("T", udf_obssize(col("obs")))
      .withColumn("obslik", udf_multinomialprob(col("obs"), col("M"), col("k"), col("T"), col("B")))
      .withColumn("prob", udf_fwd(col("M"), col("T"), col("Pi"), col("A"), col("obslik")))
  }

  val udf_toarray: UserDefinedFunction = udf((s: String) => s.split(";").map(_.toInt))
  val udf_obssize: UserDefinedFunction = udf((s: Seq[Int]) => s.length)

  val udf_gamma: UserDefinedFunction = udf((input: Row) => input.get(0).asInstanceOf[Seq[Double]])
  val udf_gamma_str: UserDefinedFunction = udf((input: Row) => input.get(0).asInstanceOf[Seq[Double]].mkString(","))

  val udf_loglik: UserDefinedFunction = udf((input: Row) => input.get(1).asInstanceOf[Double])

  val udf_xi_summed: UserDefinedFunction = udf((input: Row) => input.get(2).asInstanceOf[Seq[Double]])
  val udf_xi_summed_str: UserDefinedFunction = udf((input: Row) => input.get(2).asInstanceOf[Seq[Double]].mkString(","))

  val udf_exp_num_visits1: UserDefinedFunction = udf((input: Seq[Double], M: Int, T: Int) => {
    val gamma = new DenseMatrix(M, T, input.toArray)
    gamma(::, 1).toArray
  })

  val udf_exp_num_visits12: UserDefinedFunction = udf((input: Seq[Double], M: Int, T: Int) => {
    val output = Array.empty[Double]
    (0 until M).foreach(i => output :+ input(i))
    output
  })

  val udf_exp_num_emit: UserDefinedFunction = udf((input: Seq[Double], M: Int, k: Int, T: Int, obsin: Seq[Int]) => {
    val gamma = new DenseMatrix(M, T, input.toArray)
    var exp_num_emit = DenseMatrix.ones[Double](M, k)
    val obs = obsin.toArray
    if (T < k) {
      (0 until T).foreach(t => {
        exp_num_emit(::, obs(t)) := exp_num_emit(::, obs(t)) + gamma(::, t)
      })
    } else {
      (0 until k).foreach(o => {
        val ndx = obs.zipWithIndex.filter(_._1 == o).map(_._2)
        if (ndx.length > 0) {
          val cont = DenseVector.zeros[Double](M)
          ndx.foreach(i => cont := cont + gamma(::, i))
          exp_num_emit(::, o) := exp_num_emit(::, o) + cont
        }
      })
    }
    exp_num_emit.toArray
  })

  val udf_multinomialprob: UserDefinedFunction = udf((obs: Seq[Int], M: Int, k: Int, T: Int, B: Seq[Double]) => {
    val funB: DenseMatrix[Double] = new DenseMatrix(M, k, B.toArray)
    val output: DenseMatrix[Double] = DenseMatrix.tabulate(M, T){case(m, t) => funB(m, obs(t))}
    //val output: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, T)
    //(0 until T).foreach(t => output(::, t) := funB(::, obs(t)))
    output.toArray
  })

  val udf_multinomialprob2: UserDefinedFunction = udf((obs: Seq[Int], M: Int, k: Int, T: Int, B: Seq[Double]) => {
    val output = Array.empty[Double]
    (0 until T).foreach(j => {
      val Mj = M * j
      (0 until M).foreach(i => output :+ B(Mj  + i ))
    })
    output
  })

  val udf_fwd: UserDefinedFunction = udf((M: Int, T: Int, Pi: Seq[Double], A: Seq[Double], obslik: Seq[Double]) => {
    val funPi: DenseVector[Double] = new DenseVector(Pi.toArray)
    val funA = new DenseMatrix(M, M, A.toArray)
    val funObslik = new DenseMatrix(M, T, obslik.toArray)

    val alpha: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, T)
    //Forwards
    alpha(::, 0) := funPi :* funObslik(::, 0)
    (1 until T).foreach(t => alpha(::, t) := (funA.t * alpha(::, t - 1)) :* funObslik(::, t))
    sum(alpha(::, T - 1))
  })

  val udf_fwdback: UserDefinedFunction = udf((M: Int, T: Int, Pi: Seq[Double], A: Seq[Double], obslik: Seq[Double]) => {
    val funPi: DenseVector[Double] = new DenseVector(Pi.toArray)
    val funA: DenseMatrix[Double] = new DenseMatrix(M, M, A.toArray)
    val funObslik: DenseMatrix[Double] = new DenseMatrix(M, T, obslik.toArray)

    val scale: DenseVector[Double] = DenseVector.ones[Double](T)
    val alpha: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, T)

    //Forwards
    alpha(::, 0) := Utils.normalise(funPi :* funObslik(::, 0), scale, 0)
    (1 until T).foreach(t => alpha(::, t) := Utils.normalise((funA.t * alpha(::, t - 1)) :* funObslik(::, t), scale, t))
    val loglik: Double = sum(scale.map(Math.log(_)))

    //Backwards
    val gamma: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, T)
    val beta: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, T)
    var xi_summed: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, M)

    beta(::, T - 1) := 1.0
    gamma(::, T - 1) := normalize(alpha(::, T - 1) :* beta(::, T - 1), 1.0)

    for (t <- T - 2 to 0 by -1) {
      val b: DenseVector[Double] = beta(::, t + 1) :* funObslik(::, t + 1)
      beta(::, t) := normalize(funA * b, 1.0)
      gamma(::, t) := normalize(alpha(::, t) :* beta(::, t), 1.0)
      xi_summed = xi_summed + Utils.normalise(funA :* (alpha(::, t) * b.t))
    }
    (gamma.toArray, loglik, xi_summed.toArray)
  })

  val udf_fwdback2: UserDefinedFunction = udf((M: Int, T: Int, Pi: Seq[Double], A: Seq[Double], obslik: Seq[Double]) => {
    val funPi: DenseVector[Double] = new DenseVector(Pi.toArray)
    val funA: DenseMatrix[Double] = new DenseMatrix(M, M, A.toArray)
    val funObslik: DenseMatrix[Double] = new DenseMatrix(M, T, obslik.toArray)

    val scale: DenseVector[Double] = DenseVector.ones[Double](T)
    val alpha: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, T)

    //Forwards
    alpha(::, 0) := Utils.normalise(funPi :* funObslik(::, 0), scale, 0)
    (1 until T).foreach(t => alpha(::, t) := Utils.normalise((funA.t * alpha(::, t - 1)) :* funObslik(::, t), scale, t))
    val loglik: Double = sum(scale.map(Math.log(_)))

    //Backwards
    val gamma: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, T)
    val beta: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, T)
    var xi_summed: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, M)

    beta(::, T - 1) := 1.0
    gamma(::, T - 1) := normalize(alpha(::, T - 1) :* beta(::, T - 1), 1.0)

    for (t <- T - 2 to 0 by -1) {
      val b: DenseVector[Double] = beta(::, t + 1) :* funObslik(::, t + 1)
      beta(::, t) := normalize(funA * b, 1.0)
      gamma(::, t) := normalize(alpha(::, t) :* beta(::, t), 1.0)
      xi_summed = xi_summed + Utils.normalise(funA :* (alpha(::, t) * b.t))
    }
    (gamma.toArray, loglik, xi_summed.toArray)
  })
}
