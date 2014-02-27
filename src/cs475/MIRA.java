package cs475;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;

public class MIRA extends Predictor implements Serializable {
	private static final long serialVersionUID = 1L;

	List<Instance> _instances;
	int _online_training_iterations;
	HashMap<Integer, Double> _w = new HashMap<Integer, Double>();
	
	public MIRA(List<Instance> instances, int online_training_iterations) {
		this._instances = instances;
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
				double x2 = 0;
				for (int key : tmpmap.keySet()) {
					if (w.containsKey(key)) {
						wx += tmpmap.get(key)*w.get(key);
					}
					x2 += Math.pow(tmpmap.get(key), 2);
				}
				// Check based on the hinge-loss.
				if (yi*wx < 1) {
					for (int key : tmpmap.keySet()) {
						double wi = 0;
						if (w.containsKey(key)) {
							wi = w.get(key);
						}
						wi += (1 - yi*wx)/x2*yi*tmpmap.get(key);
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
