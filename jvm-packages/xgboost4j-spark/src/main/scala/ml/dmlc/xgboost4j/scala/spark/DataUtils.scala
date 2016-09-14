/*
 Copyright (c) 2014 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

package ml.dmlc.xgboost4j.scala.spark

import scala.collection.JavaConverters._
import org.apache.spark.mllib.linalg.{SparseVector, DenseVector, Vector}
import org.apache.spark.mllib.regression.{LabeledPoint => SparkLabeledPoint}
import ml.dmlc.xgboost4j.LabeledPoint
import org.apache.spark.sql.DataFrame
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.types.StructField

object DataUtils extends Serializable {
  implicit def fromSparkPointsToXGBoostPointsJava(sps: Iterator[SparkLabeledPoint])
    : java.util.Iterator[LabeledPoint] = {
    fromSparkPointsToXGBoostPoints(sps).asJava
  }

  implicit def fromSparkPointsToXGBoostPoints(sps: Iterator[SparkLabeledPoint]):
      Iterator[LabeledPoint] = {
    for (p <- sps) yield {
      p.features match {
        case denseFeature: DenseVector =>
          LabeledPoint.fromDenseVector(p.label.toFloat, denseFeature.values.map(_.toFloat))
        case sparseFeature: SparseVector =>
          LabeledPoint.fromSparseVector(p.label.toFloat, sparseFeature.indices,
            sparseFeature.values.map(_.toFloat))
      }
    }
  }

  implicit def fromSparkVectorToXGBoostPointsJava(sps: Iterator[Vector])
    : java.util.Iterator[LabeledPoint] = {
    fromSparkVectorToXGBoostPoints(sps).asJava
  }
  implicit def fromSparkVectorToXGBoostPoints(sps: Iterator[Vector])
    : Iterator[LabeledPoint] = {
    for (p <- sps) yield {
      p match {
        case denseFeature: DenseVector =>
          LabeledPoint.fromDenseVector(0.0f, denseFeature.values.map(_.toFloat))
        case sparseFeature: SparseVector =>
          LabeledPoint.fromSparseVector(0.0f, sparseFeature.indices,
            sparseFeature.values.map(_.toFloat))
      }
    }
  }

  implicit def dataframeToLabledPoints(dataset: DataFrame, labelColumn: String = "label",
    featuresColumn: String = "features"): RDD[SparkLabeledPoint] = {
      dataset.select(labelColumn, featuresColumn).rdd map { row =>
        new SparkLabeledPoint(row.getDouble(0), row.getAs[Vector](1))}
  }

  def appendOutput(df: DataFrame, colName: String, colType: DataType,
      values: RDD[Array[Array[Float]]]): DataFrame = {

    val dfRDD = df.rdd.zipWithIndex() map {x => (x._2, x._1) }
    val dataRDD = values.zipWithIndex() map {x => (x._2, x._1) }
    val rows = dfRDD.join(dataRDD) map { case(id, (row, value)) =>
      Row.fromSeq(row.toSeq :+ value)}
    df.sqlContext.createDataFrame(rows, StructType(
      df.schema.fields :+ StructField(colName, colType, true)))
  }
}
