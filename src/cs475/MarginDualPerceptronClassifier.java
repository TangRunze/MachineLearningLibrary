package cs475;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;

public class MarginDualPerceptronClassifier extends Predictor implements Serializable {
	private static final long serialVersionUID = 1L;
	
	List<Instance> _instances;
	String _kernelString;
	int _polynomial_kernel_exponent;
	int _online_training_iterations;
	double[] _alpha;
	KernelFun _kernel = null;
	
	public MarginDualPerceptronClassifier(List<Instance> instances, String kernelString, int polynomial_kernel_exponent, int online_training_iterations) {
		this._instances = instances;
		this._kernelString = kernelString;
		this._polynomial_kernel_exponent = polynomial_kernel_exponent;
		this._online_training_iterations = online_training_iterations;
	}

	@Override
	public void train(List<Instance> instances) {
		// Calculate the number of instances in the data.
		int num_instance = 0;
		for (Instance instance : instances) {
			num_instance++;
		}
		
		double[][] G = new double[num_instance][num_instance];
		
		
		if (this._kernelString.equalsIgnoreCase("linear")) {
			this._kernel = new LinearKernel();
		} else if (this._kernelString.equalsIgnoreCase("polynomial")) {
			this._kernel = new PolynomialKernel();
		}
		
		// Calculate the Gram Matrix.
		int i = 0;
		for (Instance instance1 : instances) {
			HashMap<Integer, Double> tmpmap1 = new HashMap<Integer, Double>();
			tmpmap1 = instance1.getFeatureVector().getMap();
			int j = 0;
			for (Instance instance2 : instances) {
				HashMap<Integer, Double> tmpmap2 = new HashMap<Integer, Double>();
				tmpmap2 = instance2.getFeatureVector().getMap();

				G[i][j] += this._kernel.calculateKernel(tmpmap1, tmpmap2, this._polynomial_kernel_exponent);
				G[j][i] = G[i][j];
				
				j++;
				
				if (j > i) {
					break;
				}
			}
			i++;
		}
		
		double[] alpha = new double[num_instance]; 
		for (int iter = 1; iter <= this._online_training_iterations; iter++) {
			i = 0;
			for (Instance instance1 : instances) {
				int yi = Integer.parseInt(instance1.getLabel().toString());
				// Our algorithm will set the label be {-1,1}.
				if (yi == 0) {
					yi = -1;
				}
				int j = 0;
				double yhat = 0;
				for (Instance instance2 : instances) {
					int yj = Integer.parseInt(instance2.getLabel().toString());
					// Our algorithm will set the label be {-1,1}.
					if (yj == 0) {
						yj = -1;
					}
					
					yhat += alpha[j]*yj*G[j][i];
					
					j++;
				}
				
				if (yi*yhat < 1) {
					alpha[i] += 1;
				}
				i++;
			}
		}
		
		this._alpha = alpha;
		this._instances = instances;
	}

	@Override
	public Label predict(Instance instance0) {
		
		HashMap<Integer, Double> map0 = instance0.getFeatureVector().getMap();
		int i = 0;
		double yhat = 0;
		for (Instance instance : this._instances) {
			int yi = Integer.parseInt(instance.getLabel().toString());
			// Our algorithm will set the label be {-1,1}.
			if (yi == 0) {
				yi = -1;
			}
			yhat += this._alpha[i]*yi*this._kernel.calculateKernel(instance.getFeatureVector().getMap(), map0, this._polynomial_kernel_exponent);
			i++;
		}

		Label label = null;
		if (yhat >= 0) {
			label = new ClassificationLabel(1);
		} else {
			label = new ClassificationLabel(0);
		}
		return label;
	}	
}
