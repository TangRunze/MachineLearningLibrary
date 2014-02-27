package cs475;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;

public class MarginPerceptronClassifier extends Predictor implements Serializable {
	private static final long serialVersionUID = 1L;
	
	List<Instance> _instances;
	double _online_learning_rate;
	int _online_training_iterations;
	HashMap<Integer, Double> _w = new HashMap<Integer, Double>();
	
	public MarginPerceptronClassifier(List<Instance> instances, double online_learning_rate, int online_training_iterations) {
		this._instances = instances;
		this._online_learning_rate = online_learning_rate;
		this._online_training_iterations = online_training_iterations;
	}

	@Override
	public void train(List<Instance> instances) {
		
		HashMap<Integer, Double> w = new HashMap<Integer, Double>();
		
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
					if (w.containsKey(key)) {
						wx += tmpmap.get(key)*w.get(key);
					}
				}
				if (yi*wx < 1) {
					for (int key : tmpmap.keySet()) {
						double wi = this._online_learning_rate*yi*tmpmap.get(key);
						if (w.containsKey(key)) {
							wi += w.get(key);
						}
						w.put(key, wi);
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
			if (this._w.containsKey(key)) {
				wx += tmpmap.get(key)*this._w.get(key);
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
