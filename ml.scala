// Databricks notebook source exported at Fri, 25 Nov 2016 00:40:59 UTC
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.feature.VectorAssembler
import spark.implicits._
import org.apache.spark.sql.types.{StructType, StructField, IntegerType,StringType,DoubleType}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.mllib.stat.Statistics


case class Types(fe0:Integer ,fe1: Double, fe2: Double, fe3: Double,fe4:Double,label:Double)
case class Type(fe0:Integer ,fe1: Double, fe2: Double, fe3: Double,fe4:Double)

//Loading the Train Dataset
val df=sc.textFile("/FileStore/tables/hcwwa1f71479189070711/train.csv").mapPartitions(_.drop(1)).map(line => line.split(",")).map(e => Types(e(0).toInt,e(1).toDouble,e(2).toDouble,e(3).toDouble,e(4).toDouble,e(5).toDouble)).toDF()
/*
//Loading the test Dataset
val df_test=sc.textFile("/FileStore/tables/hcwwa1f71479189070711/test.csv").mapPartitions(_.drop(1)).map(line => line.split(",")).map(e => Type(e(0).toInt,e(1).toDouble,e(2).toDouble,e(3).toDouble,e(4).toDouble)).toDF()
*/
/// Feature annalysis////DIMENSIONALITY REDUCTION

val feat1 = df.select($"fe1").rdd.map(_.getDouble(0))
val feat2 = df.select($"fe2").rdd.map(_.getDouble(0))
val feat3 = df.select($"fe3").rdd.map(_.getDouble(0))
val feat4 = df.select($"fe4").rdd.map(_.getDouble(0))

val f12: Double = Statistics.corr(feat1, feat2, "spearman") 
val f13: Double = Statistics.corr(feat1, feat3, "spearman")
val f14: Double = Statistics.corr(feat1, feat4, "spearman")
val f23: Double = Statistics.corr(feat2, feat3, "spearman")
val f24: Double = Statistics.corr(feat2, feat4, "spearman")
val f34: Double = Statistics.corr(feat3, feat4, "spearman")


println("features correlation")
println ("f12="+f12)
println ("f13="+f13)
println ("f14="+f14)
println ("f23="+f23)
println ("f24="+f24)
println ("f34="+f34)


//

// Preprocessing the Train Data
val dataset = new VectorAssembler().setInputCols(Array("fe1","fe2","fe4")).setOutputCol("features_original").transform(df)
val scaled_dataset = new MinMaxScaler().setInputCol("features_original").setOutputCol("features").setMax(1).setMin(0).fit(dataset).transform(dataset)
//val scaled_dataset = new VectorAssembler().setInputCols(Array("fe1","fe2","fe4")).setOutputCol("features").transform(df)

/*
//preprocessing the test dataset
var dataset2 = new VectorAssembler().setInputCols(Array("fe1","ef2","fe3","fe4")).setOutputCol("features_original").transform(df_test)
var scaled_test = new MinMaxScaler().setInputCol("features_original").setOutputCol("features").setMax(1).setMin(-1).fit(dataset).transform(dataset2)
val test = scaler2
*/

//Creating training and Validation sets
val Array(train, test) = scaled_dataset.randomSplit(Array(0.8, 0.20))
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label")


//////////// RANDOM FOREST MODEL ////////////////////////////////
println("//////////// RANDOM FOREST MODEL ////////////////////////////////")
println ("CREATING THE RANDOM FOREST MODEL")
val max_depth = List(2,3,4,5,6);
val num_trees = List(5,10,15,20);
var acc_rf: Double = Double.MinValue
var md , nt,f1,recall,precision,f1_rf,wp_rf,wc_rf =0.0;

for(d <- max_depth ; t <- num_trees)
{
  val rf = new RandomForestClassifier().setNumTrees(t).setMaxDepth(d);
  val rf_model= rf.fit(train);
  val predictions = rf_model.transform(test);
  val accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
  if(accuracy>acc_rf)
  {
    acc_rf=accuracy;
    val final_rf_model = rf_model ;
    f1_rf=evaluator.setMetricName("f1").evaluate(predictions);
    wc_rf=evaluator.setMetricName("weightedRecall").evaluate(predictions);
    wp_rf=evaluator.setMetricName("weightedPrecision").evaluate(predictions);
    md=d;
    nt=t;
  }
  println("The accuracy of random forest for parameters max_depth="+d+"; num_trees="+t+" = "+accuracy)
  
  //predictions.show();
}
println()
println("The final random forest model selected has parameters max_depth="+md+"; num_trees="+nt)
println("This selected rf model has accuracy="+acc_rf+" F Measure=" +f1_rf+" Weighted Recall="+wc_rf+" weighted Precision"+wp_rf)
println()


///////////// MULTILAYER PERCEPTRON //////////////////
println("//////////// MULTILAYER PERCEPTRON MODEL ////////////////////////////////")
///// the first layer is equal to the number of features while the last layer equals the number of classes
println(" 1 Hidden layer ANN")
println("The accuracies for various parameters of the hidden layer are")

var acc_ann: Double = Double.MinValue
var layers = Array[Int](3, 2, 2)
var hidden_number = List(1,2,3,4,5,6,7,8,9,10);
var hidden_nodes,f1_ann,wp_ann,wc_ann =0.0;
/////// 1 layer perceptron/////////////////
for(n <- hidden_number)
{
  layers= Array[Int](3, n, 2);
  val ann = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
  
  val ann_model = ann.fit(train)
  
  val ann_prediction = ann_model.transform(test)
  //ann_prediction.show()
  val acc_nn = evaluator.setMetricName("accuracy").evaluate(ann_prediction)
  if(acc_nn > acc_ann)
  {
    acc_ann=acc_nn;
    val final_ann_model = ann_model ;
    f1_ann=evaluator.setMetricName("f1").evaluate(ann_prediction);
    wc_ann=evaluator.setMetricName("weightedRecall").evaluate(ann_prediction);
    wp_ann=evaluator.setMetricName("weightedPrecision").evaluate(ann_prediction);
    
    hidden_nodes = n ;
    
  }
  println(acc_nn)
}
println("The best ANN model with 1 hidden layer gave accuracy="+acc_ann+"and has "+hidden_nodes+" in its hidden layer")
println("This ann model has accuracy="+acc_ann+" F Measure=" +f1_ann+" Weighted Recall="+wc_ann+" weighted Precision"+wp_ann)
println()
println(" 2 hidden layer ANN")
println("The accuracies for various parameters of the hidden layers are")
//////// 2 layer perceptron ///////////////////////
var hidden1,hidden2,f1_2ann,wc_2ann,wp_2ann=0.0;
var acc_2ann: Double = Double.MinValue

for(n1 <- hidden_number)
{
  for(n2 <- hidden_number )
  {
    layers= Array[Int](3, n1,n2, 2);
    val ann = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
  
    val ann_model = ann.fit(train)
  
    val ann_prediction = ann_model.transform(test)
    //ann_prediction.show()
    val acc_2nn = evaluator.setMetricName("accuracy").evaluate(ann_prediction)
    if(acc_2nn > acc_2ann)
    {
      acc_2ann=acc_2nn;
      val final_ann_model = ann_model ;
      f1_2ann=evaluator.setMetricName("f1").evaluate(ann_prediction);
      wc_2ann=evaluator.setMetricName("weightedRecall").evaluate(ann_prediction);
      wp_2ann=evaluator.setMetricName("weightedPrecision").evaluate(ann_prediction);
      
      hidden1 = n1 ;
      hidden2=n2
    
    }
    println(acc_2nn)
  }
}
println("The best ANN model with 2 hidden layers gave accuracy="+acc_2ann+"and has "+hidden1+" in its 1st hidden layer and "+hidden2+" nodes in 2nd")
if(acc_2ann>acc_ann)
{
    println("Best ann model has 2 hidden layers and accuracy="+acc_2ann+" F Measure=" +f1_2ann+" Weighted Recall="+wc_2ann+" weighted Precision"+wp_2ann)
}
else
{
  println("Best ann model has 1 hidden layer and accuracy="+acc_ann+" F Measure=" +f1_ann+" Weighted Recall="+wc_ann+" weighted Precision"+wp_ann)
}

println()


/////////////////////////BOOSTING ////////////////////////
println()
println("//////////// GRADIENT BOOSTED TREE MODEL ////////////////////////////////")

val trees= List(2,4,8,10,12,14);
val dept =List(2,3,4);
var acc_gb: Double = Double.MinValue
var no_tr,depth_gb,f1_gb,wc_gb,wp_gb=0.0;

for(tr <-trees ; dep<-dept)
{
  val gb = new GBTClassifier().setMaxIter(tr).setMaxDepth(dep)
  val gb_model = gb.fit(train)
  val gb_prediction = gb_model.transform(test)
  //predictions.show()
  val acc_g = evaluator.setMetricName("accuracy").evaluate(gb_prediction)
  if(acc_g > acc_gb)
    {
      acc_gb=acc_g;
      val final_gb_model = gb_model ;
      f1_gb=evaluator.setMetricName("f1").evaluate(gb_prediction);
      wc_gb=evaluator.setMetricName("weightedRecall").evaluate(gb_prediction);
      wp_gb=evaluator.setMetricName("weightedPrecision").evaluate(gb_prediction);
      
      no_tr=tr;
      depth_gb=dep;
    
    }
  
  println("GB Model with parameters #trees="+tr+" ;#maxtreedepth="+dep+" has accuracy="+acc_g) 
  
}
println()
println("The best GB model has parameters #trees="+no_tr+" #maxdepth="+depth_gb)
println("This GB model has accuracy="+acc_gb+" F Measure=" +f1_gb+" Weighted Recall="+wc_gb+" weighted Precision"+wp_gb)
println()

/// END OF PROGRAM

