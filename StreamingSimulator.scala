import org.apache.spark.sql.SparkSession

/**
 * ═══════════════════════════════════════════════════════════════
 *  Intelligent Big Data Analytics System
 *  Streaming Data Simulator
 *
 *  Purpose:
 *    Simulates near real-time customer data arriving by
 *    splitting the Telco dataset into small CSV files and
 *    writing them one-by-one into data/streaming/ with a delay.
 *    Phase 2 (StreamingPredictor) watches that directory and
 *    processes each file as a micro-batch.
 *
 *  Run this in a SEPARATE terminal while Phase2 is running.
 * ═══════════════════════════════════════════════════════════════
 */
object StreamingSimulator {

  val BATCH_SIZE  = 50       // customers per micro-batch file
  val DELAY_MS    = 8000     // 8 seconds between batches (matches trigger)
  val INPUT_CSV  = "C:/ChurnProject/ChurnPredictionSystem/data/telco_churn.csv.csv"
  val OUTPUT_DIR = "C:/ChurnProject/ChurnPredictionSystem/data/streaming"

  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:\\hadoop")

    val spark = SparkSession.builder()
      .appName("StreamingSimulator")
      .master("local[2]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    println("\n" + "=" * 65)
    println("  Streaming Data Simulator")
    println(s"  Batch size: $BATCH_SIZE rows  |  Interval: ${DELAY_MS / 1000}s")
    println("=" * 65)

    // Load the full dataset and drop the Churn label column
    // (streaming data arrives without labels — model predicts them)
    val fullDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(INPUT_CSV)
      .drop("Churn")

    val rows      = fullDF.collect()
    val totalRows = rows.length
    println(s"\n[SIM] Total records available for streaming: $totalRows")

    var batchNum = 0

    rows.grouped(BATCH_SIZE).foreach { batchRows =>

      val batchDF = spark.createDataFrame(
        spark.sparkContext.parallelize(batchRows),
        fullDF.schema
      )

      // Write each batch as a single CSV file into the streaming directory
      val batchPath = s"$OUTPUT_DIR/batch_${"%04d".format(batchNum)}"
      batchDF.coalesce(1)
        .write
        .option("header", "true")
        .mode("overwrite")
        .csv(batchPath)

      println(s"[SIM] Micro-batch $batchNum written → $batchPath  (${batchRows.length} rows)")
      batchNum += 1

      // Wait before sending next batch
      Thread.sleep(DELAY_MS)
    }

    println(s"\n[SIM] All $batchNum batches streamed. Simulation complete.")
    spark.stop()
  }
}
