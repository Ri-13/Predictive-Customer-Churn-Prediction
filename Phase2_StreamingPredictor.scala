import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.streaming.{StreamingQueryListener, Trigger}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.functions.vector_to_array

object Phase2_StreamingPredictor {

  val MODEL_PATH      = "C:/ChurnProject/ChurnPredictionSystem/model/churn_rf_pretrained"
  val STREAM_INPUT    = "C:/ChurnProject/ChurnPredictionSystem/data/streaming/"
  val STREAM_OUTPUT   = "C:/ChurnProject/ChurnPredictionSystem/output/streaming_results/"
  val CHECKPOINT_PATH = "C:/ChurnProject/ChurnPredictionSystem/checkpoints/streaming/"
  val TRIGGER_SECS    = "10 seconds"

  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:\\hadoop")

    val spark = SparkSession.builder()
      .appName("Phase2_StreamingPredictor_ChurnPrediction")
      .master("local[*]")
      .config("spark.sql.shuffle.partitions", "4")
      .config("spark.driver.memory", "4g")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    println("\n" + "=" * 65)
    println("  Intelligent Big Data Analytics System")
    println("  PHASE 2 — Near Real-Time Streaming Prediction")
    println("=" * 65)

    // ─────────────────────────────────────────────
    // STEP 1: Load Pre-Trained Model
    // ─────────────────────────────────────────────
    println(s"\n[STREAM] Loading pre-trained model from: $MODEL_PATH")
    val preTrainedModel: PipelineModel = PipelineModel.load(MODEL_PATH)
    println("[STREAM] ✓ Pre-trained model loaded successfully.")

    // ─────────────────────────────────────────────
    // STEP 2: Define Schema
    // ─────────────────────────────────────────────
    val streamSchema = StructType(Seq(
      StructField("customerID",       StringType,  nullable = true),
      StructField("gender",           StringType,  nullable = true),
      StructField("SeniorCitizen",    IntegerType, nullable = true),
      StructField("Partner",          StringType,  nullable = true),
      StructField("Dependents",       StringType,  nullable = true),
      StructField("tenure",           IntegerType, nullable = true),
      StructField("PhoneService",     StringType,  nullable = true),
      StructField("MultipleLines",    StringType,  nullable = true),
      StructField("InternetService",  StringType,  nullable = true),
      StructField("OnlineSecurity",   StringType,  nullable = true),
      StructField("OnlineBackup",     StringType,  nullable = true),
      StructField("DeviceProtection", StringType,  nullable = true),
      StructField("TechSupport",      StringType,  nullable = true),
      StructField("StreamingTV",      StringType,  nullable = true),
      StructField("StreamingMovies",  StringType,  nullable = true),
      StructField("Contract",         StringType,  nullable = true),
      StructField("PaperlessBilling", StringType,  nullable = true),
      StructField("PaymentMethod",    StringType,  nullable = true),
      StructField("MonthlyCharges",   DoubleType,  nullable = true),
      StructField("TotalCharges",     DoubleType,  nullable = true)
    ))

    // ─────────────────────────────────────────────
    // STEP 3: Read Streaming Data
    // ─────────────────────────────────────────────
    val rawStreamDF = spark.readStream
      .option("header", "true")
      .option("maxFilesPerTrigger", 1)
      .schema(streamSchema)
      .csv(STREAM_INPUT)

    println(s"[STREAM] Watching input directory: $STREAM_INPUT")
    println(s"[STREAM] Micro-batch trigger interval: $TRIGGER_SECS")

    // ─────────────────────────────────────────────
    // STEP 4: Preprocessing
    // ─────────────────────────────────────────────
    val cleanStreamDF = rawStreamDF
      .na.fill(0.0, Seq("TotalCharges", "MonthlyCharges"))
      .na.fill(0,   Seq("SeniorCitizen", "tenure"))
      .na.fill("No")

    // ─────────────────────────────────────────────
    // STEP 5: Apply Pre-Trained Model
    // ─────────────────────────────────────────────
    val predictions = preTrainedModel.transform(cleanStreamDF)

    // ─────────────────────────────────────────────
    // STEP 6: Enrich Output
    // ─────────────────────────────────────────────
    val enrichedDF = predictions
      .withColumn("churn_probability",
        round(vector_to_array(col("probability"))(1), 4)
      )
      .withColumn("churn_prediction",
        when(col("prediction") === 1.0, "WILL CHURN")
          .otherwise("WILL RETAIN"))
      .withColumn("risk_tier",
        when(col("churn_probability") >= 0.75, "HIGH RISK")
          .when(col("churn_probability") >= 0.50, "MEDIUM RISK")
          .otherwise("LOW RISK"))
      .withColumn("processed_at", current_timestamp())
      .select(
        col("customerID"),
        col("tenure"),
        col("Contract"),
        col("MonthlyCharges"),
        col("InternetService"),
        col("churn_probability"),
        col("churn_prediction"),
        col("risk_tier"),
        col("processed_at")
      )

    // ─────────────────────────────────────────────
    // STEP 7a: Console Sink
    // ─────────────────────────────────────────────
    val consoleQuery = enrichedDF.writeStream
      .outputMode("append")
      .format("console")
      .option("truncate", "false")
      .option("numRows", 20)
      .trigger(Trigger.ProcessingTime(TRIGGER_SECS))
      .queryName("console_sink")
      .start()

    // ─────────────────────────────────────────────
    // STEP 7b: CSV File Sink
    // ─────────────────────────────────────────────
    val fileQuery = enrichedDF.writeStream
      .outputMode("append")
      .format("csv")
      .option("header", "true")
      .option("path", STREAM_OUTPUT)
      .option("checkpointLocation", CHECKPOINT_PATH)
      .trigger(Trigger.ProcessingTime(TRIGGER_SECS))
      .queryName("file_sink")
      .start()

    println(s"\n[STREAM] ✓ Streaming queries started.")
    println(s"[STREAM]   Results written to: $STREAM_OUTPUT")
    println(s"[STREAM]   Drop CSV files into $STREAM_INPUT to trigger predictions.")
    println(s"[STREAM]   Press Ctrl+C to stop.\n")

    // ─────────────────────────────────────────────
    // STEP 8: Streaming Query Listener
    // ─────────────────────────────────────────────
    spark.streams.addListener(new StreamingQueryListener {

      override def onQueryStarted(event: StreamingQueryListener.QueryStartedEvent): Unit =
        println(s"[MONITOR] Query started — name: ${event.name}  id: ${event.id}")

      override def onQueryProgress(event: StreamingQueryListener.QueryProgressEvent): Unit = {
        val p = event.progress

        val triggerTime = Option(p.durationMs.get("triggerExecution"))
          .map(_.toString.toLong)
          .getOrElse(0L)

        if (p.numInputRows > 0) {
          println(
              f"[MONITOR] Batch ${p.batchId}%3d | " +
              f"Input rows: ${p.numInputRows}%4d | " +
              f"Processing rate: ${p.processedRowsPerSecond}%7.1f rows/sec | " +
              f"Trigger: ${triggerTime}%d ms"
          )
        } else {
          println(
              f"[MONITOR] Batch ${p.batchId}%3d | " +
              f"No new data | Trigger: ${triggerTime}%d ms"
          )
        }
      }

      override def onQueryTerminated(event: StreamingQueryListener.QueryTerminatedEvent): Unit =
        println(s"[MONITOR] Query terminated — id: ${event.id}  " +
          s"exception: ${event.exception.getOrElse("none")}")
    })

    spark.streams.awaitAnyTermination()
  }
}