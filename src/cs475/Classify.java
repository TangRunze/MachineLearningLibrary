package cs475;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.LinkedList;
import java.util.List;

import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;

public class Classify {
	static public LinkedList<Option> options = new LinkedList<Option>();
	private static double gd_eta;
	private static int gd_iterations;
	private static int num_features_to_select;
	private static double online_learning_rate;
	private static int online_training_iterations;
	private static int polynomial_kernel_exponent;
	
	public static void main(String[] args) throws IOException {
		// Parse the command line.
		String[] manditory_args = { "mode"};
		createCommandLineOptions();
		CommandLineUtilities.initCommandLineParameters(args, Classify.options, manditory_args);
	
		String mode = CommandLineUtilities.getOptionValue("mode");
		String data = CommandLineUtilities.getOptionValue("data");
		String predictions_file = CommandLineUtilities.getOptionValue("predictions_file");
		String algorithm = CommandLineUtilities.getOptionValue("algorithm");
		String model_file = CommandLineUtilities.getOptionValue("model_file");
		
		gd_eta = 0.01;
		if (CommandLineUtilities.hasArg("gd_eta")) {
			gd_eta = CommandLineUtilities.getOptionValueAsFloat("gd_eta");
		}
		gd_iterations = 20;
		if (CommandLineUtilities.hasArg("gd_iterations")) {
			gd_iterations = CommandLineUtilities.getOptionValueAsInt("gd_iterations");
		}
		num_features_to_select = -1;
		if (CommandLineUtilities.hasArg("num_features_to_select")) {
			num_features_to_select = CommandLineUtilities.getOptionValueAsInt("num_features_to_select");
		}
		
		online_learning_rate = 1.0;
		if (CommandLineUtilities.hasArg("online_learning_rate")) {
			online_learning_rate = CommandLineUtilities.getOptionValueAsFloat("online_learning_rate");
		}
		online_training_iterations = 5;
		if (CommandLineUtilities.hasArg("online_training_iterations")) {
			online_training_iterations = CommandLineUtilities.getOptionValueAsInt("online_training_iterations");
		}
		polynomial_kernel_exponent = 2;
		if (CommandLineUtilities.hasArg("polynomial_kernel_exponent")) {
			polynomial_kernel_exponent = CommandLineUtilities.getOptionValueAsInt("polynomial_kernel_exponent");
		}
		
		if (mode.equalsIgnoreCase("train")) {
			if (data == null || algorithm == null || model_file == null) {
				System.out.println("Train requires the following arguments: data, algorithm, model_file");
				System.exit(0);
			}
			// Load the training data.
			DataReader data_reader = new DataReader(data, true);
			List<Instance> instances = data_reader.readData();
			data_reader.close();
			
			// Train the model.
			Predictor predictor = train(instances, algorithm);
			saveObject(predictor, model_file);
			
		} else if (mode.equalsIgnoreCase("test")) {
			if (data == null || predictions_file == null || model_file == null) {
				System.out.println("Train requires the following arguments: data, predictions_file, model_file");
				System.exit(0);
			}
			
			// Load the test data.
			DataReader data_reader = new DataReader(data, true);
			List<Instance> instances = data_reader.readData();
			data_reader.close();
			
			// Load the model.
			Predictor predictor = (Predictor)loadObject(model_file);
			evaluateAndSavePredictions(predictor, instances, predictions_file);
		} else {
			System.out.println("Requires mode argument.");
		}
	}
	

	private static Predictor train(List<Instance> instances, String algorithm) {
		Predictor predictor = null;
		// TODO Train the model using "algorithm" on "data"
		if (algorithm.equalsIgnoreCase("majority")) {
			predictor = new MajorityClassifier(instances);
		} else if (algorithm.equalsIgnoreCase("even_odd")) {
			predictor = new EvenOddClassifier(instances);
		} else if (algorithm.equalsIgnoreCase("logistic_regression")) {
			predictor = new LogisticRegressionClassifier(instances, gd_eta, gd_iterations, num_features_to_select);
		} else if (algorithm.equalsIgnoreCase("margin_perceptron")) {
			predictor = new MarginPerceptronClassifier(instances, online_learning_rate, online_training_iterations);
		} else if (algorithm.equalsIgnoreCase("perceptron_linear_kernel")) {
			predictor = new MarginDualPerceptronClassifier(instances, "linear", polynomial_kernel_exponent, online_training_iterations);
		} else if (algorithm.equalsIgnoreCase("perceptron_polynomial_kernel")) {
			predictor = new MarginDualPerceptronClassifier(instances, "polynomial", polynomial_kernel_exponent, online_training_iterations);
		} else {
			System.out.println("Algorithm not found.");
		}
		
		List<Instance> instances_origin = instances;
		
		// Run feature selection.
		if (num_features_to_select > -1) {
			FeatureSelection featureselection = new FeatureSelection(instances);
			instances = featureselection.select(num_features_to_select);
		}
		
		// Train the model using "algorithm" on "data"
		predictor.train(instances);
		// TODO Evaluate the model
		AccuracyEvaluator accuracyevaluator = new AccuracyEvaluator(instances_origin, predictor);
		double accuracy = accuracyevaluator.evaluate(instances_origin, predictor);
		System.out.println("The accuracy of the training data is: " + Double.toString(accuracy));
		return predictor;
	}

	private static void evaluateAndSavePredictions(Predictor predictor,
			List<Instance> instances, String predictions_file) throws IOException {
		PredictionsWriter writer = new PredictionsWriter(predictions_file);
		// TODO Evaluate the model if labels are available. 
		AccuracyEvaluator accuracyevaluator = new AccuracyEvaluator(instances, predictor);
		double accuracy = accuracyevaluator.evaluate(instances, predictor);
		System.out.println("The accuracy of the test data is: " + Double.toString(accuracy));
		
		for (Instance instance : instances) {
			Label label = predictor.predict(instance);
			writer.writePrediction(label);
		}
		
		writer.close();
		
	}

	public static void saveObject(Object object, String file_name) {
		try {
			ObjectOutputStream oos =
				new ObjectOutputStream(new BufferedOutputStream(
						new FileOutputStream(new File(file_name))));
			oos.writeObject(object);
			oos.close();
		}
		catch (IOException e) {
			System.err.println("Exception writing file " + file_name + ": " + e);
		}
	}

	/**
	 * Load a single object from a filename. 
	 * @param file_name
	 * @return
	 */
	public static Object loadObject(String file_name) {
		ObjectInputStream ois;
		try {
			ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(new File(file_name))));
			Object object = ois.readObject();
			ois.close();
			return object;
		} catch (IOException e) {
			System.err.println("Error loading: " + file_name);
		} catch (ClassNotFoundException e) {
			System.err.println("Error loading: " + file_name);
		}
		return null;
	}
	
	public static void registerOption(String option_name, String arg_name, boolean has_arg, String description) {
		OptionBuilder.withArgName(arg_name);
		OptionBuilder.hasArg(has_arg);
		OptionBuilder.withDescription(description);
		Option option = OptionBuilder.create(option_name);
		
		Classify.options.add(option);		
	}
	
	private static void createCommandLineOptions() {
		registerOption("data", "String", true, "The data to use.");
		registerOption("mode", "String", true, "Operating mode: train or test.");
		registerOption("predictions_file", "String", true, "The predictions file to create.");
		registerOption("algorithm", "String", true, "The name of the algorithm for training.");
		registerOption("model_file", "String", true, "The name of the model file to create/load.");
		
		registerOption("gd_eta", "int", true, "The step size parameter for GD.");
		registerOption("gd_iterations", "int", true, "The number of GD iterations.");
		registerOption("num_features_to_select", "int", true, "The number of features to select.");
		
		registerOption("online_learning_rate", "double", true, "The learning rate for perceptron.");
		registerOption("online_training_iterations", "int", true, "The number of training iterations for online methods.");
		// Other options will be added here.
	}
}
