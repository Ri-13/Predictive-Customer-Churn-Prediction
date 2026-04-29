import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.PipelineModel

object Phase1_BatchTraining {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:\\hadoop")

    val spark = SparkSession.builder()
      .appName("Phase1_BatchTraining_ChurnPrediction")
      .master("local[*]")
      .config("spark.driver.memory", "4g")
      .config("spark.sql.shuffle.partitions", "8")
      .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
      .config("spark.hadoop.io.native.lib.available", "false")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    println("=================================================================")
    println("  Intelligent Big Data Analytics System")
    println("  PHASE 1 — Batch Processing & Model Training")
    println("=================================================================")

   //  Load the dataset
    val rawDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("C:/ChurnProject/ChurnPredictionSystem/data/telco_churn.csv.csv")

    println("\n[BATCH] Raw dataset loaded.")
    println(s"[BATCH] Rows: ${rawDF.count()}  |  Columns: ${rawDF.columns.length}")
    rawDF.printSchema()
    rawDF.show(5, truncate = false)

   // Preprocessing
    val medianTotalCharges = rawDF
      .filter(col("TotalCharges") =!= " ")
      .select(col("TotalCharges").cast("double"))
      .stat.approxQuantile("TotalCharges", Array(0.5), 0.001)(0)

    var df = rawDF
      .drop("customerID")
      .withColumn("TotalCharges",
        when(col("TotalCharges") === " ", medianTotalCharges.toString)
          .otherwise(col("TotalCharges")))
      .withColumn("TotalCharges", col("TotalCharges").cast("double"))
      .na.fill("No")
      .withColumn("label",
        when(col("Churn") === "Yes", 1.0).otherwise(0.0))
      .drop("Churn")

    println(s"\n[BATCH] After preprocessing — Rows: ${df.count()}")
    println("[BATCH] Class distribution:")
    df.groupBy("label").count().orderBy("label").show()

    // Class Weight Balancing for SRF
    val negCount = df.filter(col("label") === 0.0).count().toDouble
    val posCount = df.filter(col("label") === 1.0).count().toDouble
    val total    = negCount + posCount

    df = df.withColumn("classWeight",
      when(col("label") === 1.0, total / (2 * posCount))
        .otherwise(total / (2 * negCount)))

    println("[BATCH] Class weights applied for SRF.")

    // Feature Engineering
    val categoricalCols = Array(
      "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
      "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
      "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
      "PaperlessBilling", "PaymentMethod"
    )
    val numericCols = Array("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges")

    println(s"\n[BATCH] Categorical features (${categoricalCols.length}): ${categoricalCols.mkString(", ")}")
    println(s"[BATCH] Numeric features    (${numericCols.length}): ${numericCols.mkString(", ")}")

    val indexers = categoricalCols.map(c =>
      new StringIndexer().setInputCol(c).setOutputCol(s"${c}_idx").setHandleInvalid("keep"))

    val encoders = categoricalCols.map(c =>
      new OneHotEncoder().setInputCol(s"${c}_idx").setOutputCol(s"${c}_ohe"))

    val featureCols = categoricalCols.map(c => s"${c}_ohe") ++ numericCols

    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    // Train_Test Split = 80:20
    val Array(trainDF, testDF) = df.randomSplit(Array(0.8, 0.2), seed = 42)
    println(s"\n[BATCH] Train samples: ${trainDF.count()}")
    println(s"[BATCH] Test  samples: ${testDF.count()}")

    // Evaluation
    def evaluateModel(predictions: org.apache.spark.sql.DataFrame,
                      modelName: String): Unit = {
      val aucEval = new BinaryClassificationEvaluator()
        .setLabelCol("label")
        .setRawPredictionCol("rawPrediction")
        .setMetricName("areaUnderROC")
      val accEval = new MulticlassClassificationEvaluator()
        .setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
      val precEval = new MulticlassClassificationEvaluator()
        .setLabelCol("label").setPredictionCol("prediction").setMetricName("weightedPrecision")
      val recEval = new MulticlassClassificationEvaluator()
        .setLabelCol("label").setPredictionCol("prediction").setMetricName("weightedRecall")
      val f1Eval = new MulticlassClassificationEvaluator()
        .setLabelCol("label").setPredictionCol("prediction").setMetricName("f1")

      val auc  = aucEval.evaluate(predictions)
      val acc  = accEval.evaluate(predictions)
      val prec = precEval.evaluate(predictions)
      val rec  = recEval.evaluate(predictions)
      val f1   = f1Eval.evaluate(predictions)

      println(s"\n────────────────────────────────────────")
      println(s"  $modelName RESULTS")
      println(s"────────────────────────────────────────")
      println(f"  AUC-ROC   : $auc%.4f")
      println(f"  Accuracy  : $acc%.4f")
      println(f"  Precision : $prec%.4f")
      println(f"  Recall    : $rec%.4f")
      println(f"  F1 Score  : $f1%.4f")
      println(s"────────────────────────────────────────")
    }

    // Model 1 Training : Naive Bayes Baseline 1
    println("\n[BATCH] Training Naive Bayes...")

    val nb = new NaiveBayes()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setModelType("multinomial")

    val nbPipeline = new Pipeline()
      .setStages(indexers ++ encoders ++ Array(assembler, nb))

    val nbModel       = nbPipeline.fit(trainDF)
    val nbPredictions = nbModel.transform(testDF)
    evaluateModel(nbPredictions, "NAIVE BAYES")

    // Model 2 Training : Decision Tree Baseline 2
    println("\n[BATCH] Training Decision Tree...")

    val dt = new DecisionTreeClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxDepth(10)
      .setSeed(42)

    val dtPipeline = new Pipeline()
      .setStages(indexers ++ encoders ++ Array(assembler, dt))

    val dtModel       = dtPipeline.fit(trainDF)
    val dtPredictions = dtModel.transform(testDF)
    evaluateModel(dtPredictions, "DECISION TREE")

    // Model 3 Training : SRF with CV + Class Weights (Proposed)
    println("\n[BATCH] Training Scalable Random Forest with 5-fold CV...")
    println("[BATCH] Grid size: 16 parameter combinations")

    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setWeightCol("classWeight")
      .setFeatureSubsetStrategy("sqrt")
      .setMinInstancesPerNode(3)
      .setSeed(42)

    val rfPipeline = new Pipeline()
      .setStages(indexers ++ encoders ++ Array(assembler, rf))

    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.numTrees, Array(100, 200))
      .addGrid(rf.maxDepth, Array(10, 15))
      .addGrid(rf.impurity, Array("gini", "entropy"))
      .build()

    val cvEvaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")

    val cv = new CrossValidator()
      .setEstimator(rfPipeline)
      .setEvaluator(cvEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)
      .setSeed(42)

    val cvModel = cv.fit(trainDF)
    println("[BATCH] Cross-Validation complete.")

    val rfPredictions = cvModel.transform(testDF)
    evaluateModel(rfPredictions, "SRF (PROPOSED)")

    // Confusion Matrix SRF
    println("\n  Confusion Matrix (SRF):")
    rfPredictions
      .groupBy("label", "prediction")
      .count()
      .orderBy("label", "prediction")
      .show()

    // Features Importances SRF
    val bestRFModel  = cvModel.bestModel.asInstanceOf[PipelineModel]
    val rfClassifier = bestRFModel.stages.last.asInstanceOf[RandomForestClassificationModel]
    val importances  = rfClassifier.featureImportances.toArray
    val importancePairs = featureCols.zip(importances).sortBy(-_._2).take(10)

    println("\n  Top 10 Feature Importances:")
    importancePairs.foreach { case (name, imp) =>
      println("    %-35s  %.4f".format(name, imp))
    }


  // Saves Best SRF Model
    val modelSavePath =
      "C:/ChurnProject/ChurnPredictionSystem/model/churn_rf_pretrained"
    bestRFModel.write.overwrite().save(modelSavePath)
    println(s"\n[BATCH] Best SRF model saved to: $modelSavePath")

    println("\n=================================================================")
    println("  PHASE 1 COMPLETE")
    println("=================================================================")

    spark.stop()
  }
}