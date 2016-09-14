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

package ml.dmlc.xgboost4j.scala.spark.pipeline

import org.apache.spark.ml.Model
import ml.dmlc.xgboost4j.scala.spark.{ XGBoostModel => XGBModel }
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.XGBoostParams
import org.apache.spark.sql.types.StructField
import org.apache.spark.mllib.linalg.VectorUDT
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vector
import ml.dmlc.xgboost4j.scala.spark.DataUtils._

/**
 * [[http://xgboost.readthedocs.io/en/latest/model.html XGBoost]] model for classification
 * and regression.
 *
 * @param model The ml.dmlc.xgboost4j.scala.spark.XGBoostModel instance to delegate the tasks to.
 */
class XGBoostModel(override val uid: String, model: XGBModel)
    extends Model[XGBoostModel] with XGBoostParams {

  def this(model: XGBModel) = this(Identifiable.randomUID("xgb"), model)

  /**
   * Do the transformation - this means make the predictions for the given dataset.
   * @param dataset Dataset to get the predictions for.
   */
  override def transform(dataset: DataFrame): DataFrame = {
    transformSchema(dataset.schema)
    val features: RDD[Vector] = dataset.select($(featuresCol))
      .rdd.map { row => row.getAs[Vector](0) }

    val output = model.predict(features, $(useExternalCache))
    appendOutput(dataset, $(predictionCol), new VectorUDT, output)
  }

  /**
   * Returns the new XGBoostModel
   * @param extra Additional parameters for the new model.
   */
  override def copy(extra: ParamMap): XGBoostModel = copyValues(new XGBoostModel(model), extra)

  /**
   * Validate and Transform the input schema to output schema.
   * @param schema Schema for the input dataset/dataframe.
   */

  override def transformSchema(schema: StructType): StructType =
      validateAndTransformSchema(schema, false)
}
