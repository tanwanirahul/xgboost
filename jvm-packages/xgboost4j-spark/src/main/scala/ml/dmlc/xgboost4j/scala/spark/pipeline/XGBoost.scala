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

import org.apache.spark.ml.Estimator
import org.apache.spark.ml.XGBoostParams
import org.apache.spark.sql.DataFrame
import ml.dmlc.xgboost4j.scala.spark.XGBoost
import ml.dmlc.xgboost4j.scala.spark.DataUtils._
import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.param.ParamMap

/**
 * An XGBoost estimator to be fit on the training set. Returns the instance
 * of XGBoostModel as a learned model.
 */
class XGBoost(override val uid: String) extends Estimator[XGBoostModel]
  with XGBoostParams {

  def this() = this(Identifiable.randomUID("xgb"))

 /**
  * Fit the XGBoost model on the dataset provided.
  * @param dataset The training set to fit the model on.
  * @returns An instance of XGBoostModel.
  */
  override def fit(dataset: DataFrame): XGBoostModel = {

    transformSchema(dataset.schema, logging = true)
    val trainData = dataframeToLabledPoints(dataset, $(labelCol), $(featuresCol))
    val model = XGBoost.train(trainData, paramsMap, $(rounds), $(nWorkers),
      useExternalMemory = $(useExternalCache))
    copyValues(new XGBoostModel(model).setParent(this))
  }

  /**
   * Returns the new XGBoost instance.
   * @param extra Additional parameters for the new model.
   */
  override def copy(extra: ParamMap): XGBoost = copyValues(new XGBoost(uid), extra)

  /**
   * Validate and Transform the input schema to output schema.
   * @param schema Schema for the input dataset/dataframe.
   */
  override def transformSchema(schema: StructType): StructType =
      validateAndTransformSchema(schema, false)
}
