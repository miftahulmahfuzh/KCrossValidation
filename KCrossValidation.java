 import java.io.*;
 import java.util.Random;
 import weka.core.*;
 import weka.filters.Filter;
 import weka.filters.supervised.attribute.Discretize;
 import weka.classifiers.Classifier;
 import weka.classifiers.lazy.IBk;
 import weka.classifiers.Evaluation;
 
 public class KCrossValidation {

   protected static Instances load(String filename) throws Exception {
     BufferedReader reader = new BufferedReader(new FileReader(filename));
     Instances result = new Instances(reader);
     result.setClassIndex(result.numAttributes() - 1);
     reader.close();
 
     return result;
   }
 
   public static void main(String[] args) throws Exception {

     int validateMethod = Integer.parseInt(args[1]);

     if (validateMethod == 3) {
       Instances inputTest = load(args[0]);

       // Discretization Filter
       Discretize filter = new Discretize();
       filter.setInputFormat(inputTest);
       inputTest = Filter.useFilter(inputTest, filter);

       Classifier cls = (Classifier) SerializationHelper.read("ibk.model");
       System.out.println("=== Test Result ===");
       for (int i=0;i<inputTest.numInstances();i++) {
         double classResult = cls.classifyInstance(inputTest.instance(i));
         System.out.println("class " + i + " : " + classResult);
       }

       Evaluation eval = new Evaluation(inputTest);
       eval.evaluateModel(cls, inputTest);
       System.out.println(eval.toSummaryString());
       System.out.println(eval.toMatrixString());
     } else {
       Instances inputTrain = load(args[0]);
       int folds = 10;

       // Discretization Filter
       Discretize filter = new Discretize();
       filter.setInputFormat(inputTrain);
       inputTrain = Filter.useFilter(inputTrain, filter);

       Random rand = new Random(folds);
       inputTrain.randomize(rand);
       Classifier ibk = new IBk();
       Evaluation eval = new Evaluation(inputTrain);

       if (validateMethod == 1) {
         // 10 Cross Fold Validation
         for (int n=0;n<folds;n++) {
           Instances train = inputTrain.trainCV(folds, n);
           Instances test = inputTrain.testCV(folds, n);
           ibk.buildClassifier(train);
           eval.evaluateModel(ibk, test);
         }

         System.out.println(eval.toSummaryString("=== " + folds + " Cross Validation Fold ===", false));
         System.out.println(eval.toMatrixString());
       } else if (validateMethod == 2) {
         // Full Training
         ibk.buildClassifier(inputTrain);
         Instances inputTest = load("iris-fulltraining.arff");
         filter = new Discretize();
         filter.setInputFormat(inputTest);
         inputTest = Filter.useFilter(inputTest, filter);
         eval.evaluateModel(ibk, inputTest);
         
         System.out.println(eval.toSummaryString("=== Full Training ===", false));
         System.out.println(eval.toMatrixString());
       } 

       // Write Trained Data Model
       SerializationHelper.write("ibk.model", ibk);
     }
   }
 }
