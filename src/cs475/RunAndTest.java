package cs475;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class RunAndTest {
   // Note: To use, extract all the data fines into a single folder, and rename the synthetic files to
   // "synthetic_easy.train", "synthetic_easy.dev", "synthetic_hard.train", and "synthetic_hard.dev".
   // Specify that location as the DATA_DIRECTORY_PATH, and also specify location for where you want
   // your models and predictions should be stored.

   // Note: If you are on windows, don't forget that the file separator in the path below
   // should be a forward slash ('/'), not a back slash ('\').

   // Note: Be sure that the directories specified below exist; they will not be automatically created!

   // Note: Finally, make sure that the below paths end with a '/'.
   public static final String DATA_DIRECTORY_PATH = "C:/Users/trz/workspace/CS475/";
   public static final String MODEL_DIRECTORY_PATH = "C:/Users/trz/workspace/CS475/";
   public static final String PREDICTIONS_DIRECTORY_PATH = "C:/Users/trz/workspace/CS475/";
   private Algorithm algorithm;
   private ArrayList<String> train_arguments;
   private ArrayList<String> predict_arguments;
   private Dataset[] datasets_to_run;

   public static void main(String[] args) throws IOException {
      RunAndTest myRun;

      Algorithm[] hw3_algorithms = { Algorithm.MARGIN_PERCEPTRON, Algorithm.PERCEPTRON_LINEAR_KERNEL,
            Algorithm.PERCEPTRON_POLYNOMIAL_KERNEL, Algorithm.MIRA };
      for (Algorithm alg : hw3_algorithms) {
         myRun = new RunAndTest(alg);
         //myRun.setDatasetsToRun(Dataset.MOST);
         myRun.run();
      }
   }

   private RunAndTest(Algorithm algorithm) {
      this.algorithm = algorithm;
      train_arguments = new ArrayList<String>();
      predict_arguments = new ArrayList<String>();
      datasets_to_run = Dataset.ALL;
   }

   private RunAndTest(Algorithm algorithm, ArrayList<String> train_arguments, ArrayList<String> predict_arguments,
         Dataset[] datasets_to_run) {
      this.algorithm = algorithm;
      this.train_arguments = train_arguments;
      this.predict_arguments = predict_arguments;
      this.datasets_to_run = datasets_to_run;

   }

   public void run() throws IOException {
      System.out.println("\nRunning " + algorithm.FLAG + " algorithm:");
      for (Dataset dataset : datasets_to_run) {
         System.out.println("\nRunning " + dataset.NAME + " dataset.");
         System.out.println("Training...");
         train(dataset);
         for (Filetype filetype : dataset.FILETYPES) {
            System.out.println("Calculuating predictions for ." + filetype.FLAG + " data.");
            predict(dataset, filetype);
         }
      }

      System.out.println("\n\n-------------------------------------------------------\nResults for " + algorithm.FLAG + " algorithm:\n");
      for (Dataset dataset : datasets_to_run) {
         System.out.println("Accuracy on " + dataset.NAME + " dataset:");
         for (Filetype filetype : dataset.FILETYPES) {
            if (filetype.FLAG.equals("test")) {
               continue;
            }
            System.out.println(filetype.FLAG + " data: " + String.valueOf(evaluate(dataset, filetype)));
         }
         System.out.println();
      }
      System.out.println("-------------------------------------------------------");

   }

   private void train(Dataset dataset) throws IOException {
      @SuppressWarnings("unchecked")
      ArrayList<String> arguments = (ArrayList<String>) train_arguments.clone();
      arguments.add("-mode");
      arguments.add("train");

      arguments.add("-algorithm");
      arguments.add(algorithm.FLAG);

      arguments.add("-model_file");
      arguments.add(MODEL_DIRECTORY_PATH + dataset.NAME + ".model");

      arguments.add("-data");
      arguments.add(DATA_DIRECTORY_PATH + dataset.NAME + ".train");

      Classify.main(arguments.toArray(new String[arguments.size()]));
   }

   private void predict(Dataset dataset, Filetype filetype) throws IOException {
      @SuppressWarnings("unchecked")
      ArrayList<String> arguments = (ArrayList<String>) predict_arguments.clone();

      arguments.add("-mode");
      arguments.add("test");

      arguments.add("-model_file");
      arguments.add(MODEL_DIRECTORY_PATH + dataset.NAME + ".model");

      arguments.add("-data");
      arguments.add(DATA_DIRECTORY_PATH + dataset.NAME + "." + filetype.FLAG);

      arguments.add("-predictions_file");
      arguments.add(PREDICTIONS_DIRECTORY_PATH + algorithm.FLAG + "." + dataset.NAME + "." + filetype.FLAG + ".prediction");

      Classify.main(arguments.toArray(new String[arguments.size()]));
   }

   private double evaluate(Dataset dataset, Filetype filetype) throws IOException {
      int total_labels = 0;
      int correct_labels = 0;

      BufferedReader data = new BufferedReader(new FileReader(DATA_DIRECTORY_PATH + dataset.NAME + "." + filetype.FLAG));
      BufferedReader prediction = new BufferedReader(
            new FileReader(PREDICTIONS_DIRECTORY_PATH + algorithm.FLAG + "." + dataset.NAME + "." + filetype.FLAG + ".prediction"));
      String data_line = null;
      String prediction_line = null;
      while ((data_line = data.readLine()) != null && (prediction_line = prediction.readLine()) != null) {
         ++total_labels;
         if (data_line.split(" ")[0].equals(prediction_line.trim())) {
            ++correct_labels;
         }
      }
      data.close();
      prediction.close();
      return ((double) correct_labels) / total_labels;
   }

   public void set_arguments(ArrayList<String> new_arguments) {
      set_train_arguments(new_arguments);
      set_predict_arguments(new_arguments);
   }

   public void set_train_arguments(ArrayList<String> new_arguments) {
      train_arguments = new_arguments;
   }

   public void set_predict_arguments(ArrayList<String> new_arguments) {
      predict_arguments = new_arguments;
   }

   public void setDatasetsToRun(Dataset[] datasets) {
      datasets_to_run = datasets;
   }

   private enum Filetype {
      TRAIN("train"),
      DEV("dev"),
      TEST("test");

      public final String FLAG;

      private Filetype(String flag) {
         FLAG = flag;
      }

      public static final Filetype[] ALL = { TRAIN, DEV, TEST };
      public static final Filetype[] TRAIN_AND_DEV = { TRAIN, DEV };

   }

   private enum Dataset {
      SYNTHETIC_EASY("synthetic_easy", Filetype.TRAIN_AND_DEV),
      SYNTHETIC_HARD("synthetic_hard", Filetype.TRAIN_AND_DEV),
      BIO("bio", Filetype.ALL),
      FINANCE("finance", Filetype.ALL),
      NLP("nlp", Filetype.ALL),
      SPEECH("speech", Filetype.ALL),
      VISION("vision", Filetype.ALL),
      CIRCLE("circle", Filetype.ALL);

      public final String NAME;
      public final Filetype[] FILETYPES;

      private Dataset(String ext, Filetype[] filetypes) {
         NAME = ext;
         FILETYPES = filetypes;
      }

      private static final Dataset[] ALL = { SYNTHETIC_EASY, SYNTHETIC_HARD, BIO, FINANCE, NLP, SPEECH, VISION, CIRCLE };
      private static final Dataset[] MOST = { SYNTHETIC_EASY, SYNTHETIC_HARD, BIO, FINANCE, SPEECH, VISION, CIRCLE };
      private static final Dataset[] JUST_NLP = { NLP };
   }

   private enum Algorithm {
      MAJORITY("majority"),
      EVEN_ODD("even_odd"),
      LOGISTICAL_REGRESSION("logistic_regression"),
      MARGIN_PERCEPTRON("margin_perceptron"),
      PERCEPTRON_LINEAR_KERNEL("perceptron_linear_kernel"),
      PERCEPTRON_POLYNOMIAL_KERNEL("perceptron_polynomial_kernel"),
      MIRA("mira");

      public final String FLAG;

      private Algorithm(String flag) {
         FLAG = flag;
      }

   }
}