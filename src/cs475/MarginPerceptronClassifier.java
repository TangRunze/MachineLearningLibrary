package cs475;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;

public class MarginPerceptronClassifier extends Predictor implements Serializable {
	private static final long serialVersionUID = 1L;
	
	List<Instance> _instances;
	double _online_learning_rate;
	int _online_training_iterations;
	double[] _w;
	int _length_features;
	
	public MarginPerceptronClassifier(List<Instance> instances, double online_learning_rate, int online_training_iterations) {
		this._instances = instances;
		this._online_learning_rate = online_learning_rate;
		this._online_training_iterations = online_training_iterations;
	}

	@Override
	public void train(List<Instance> instances) {
		int length_features = 0;
		
		for (Instance instance : instances) {
			int tmpID = instance.getFeatureVector().getMaxID();
			if (tmpID > length_features) {
				length_features = tmpID;
			}
		}
		this._length_features = length_features;
		
		double[] w = new double[length_features];
		
		for (int iter = 1; iter <= this._online_training_iterations; iter++) {
			for (Instance instance : instances) {
				int yi = Integer.parseInt(instance.getLabel().toString());
				// Our algorithm will set the label be {-1,1}.
				if (yi == 0) {
					yi = -1;
				}
				HashMap<Integer, Double> tmpmap = new HashMap<Integer, Double>();
				tmpmap = instance.getFeatureVector().getMap();
				double wx = 0;
				for (int key : tmpmap.keySet()) {
					wx += tmpmap.get(key)*w[key-1];
				}
				if (yi*wx < 1) {
					for (int key : tmpmap.keySet()) {
						w[key-1] += this._online_learning_rate*yi*tmpmap.get(key);
					}				
				}
			}
		}
		
		this._w = w;
	}

	@Override
	public Label predict(Instance instance) {

		HashMap<Integer, Double> tmpmap = new HashMap<Integer, Double>();
		tmpmap = instance.getFeatureVector().getMap();
		double wx = 0;
		for (int key : tmpmap.keySet()) {
			if (key <= this._length_features) {
				wx += tmpmap.get(key)*this._w[key-1];
			}
		}

		Label label = null;
		if (wx >= 0) {
			label = new ClassificationLabel(1);
		} else {
			label = new ClassificationLabel(0);
		}
		return label;
	}	
}
