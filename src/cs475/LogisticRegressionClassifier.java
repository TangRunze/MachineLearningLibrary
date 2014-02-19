package cs475;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;

public class LogisticRegressionClassifier extends Predictor implements Serializable {
	private static final long serialVersionUID = 1L;
	
	List<Instance> _instances;
	double _gd_eta;
	int _gd_iterations;
	int _num_features_to_select;
	int _length_features;
	double[] _w;
	
	public LogisticRegressionClassifier(List<Instance> instances, Double gd_eta, Integer gd_iterations, Integer num_features_to_select) {
		this._instances = instances;
		this._gd_eta = gd_eta;
		this._gd_iterations = gd_iterations;
		this._num_features_to_select = num_features_to_select;
	}

	@Override
	public void train(List<Instance> instances) {
		// TODO Auto-generated method stub
		double gd_eta = this._gd_eta;
		int gd_iterations = this._gd_iterations;
		int num_features_to_select = this._num_features_to_select;
		int length_features = 0;
		
		for (Instance instance : instances) {
			int tmpID = instance.getFeatureVector().getMaxID();
			if (tmpID > length_features) {
				length_features = tmpID;
			}
		}
		this._length_features = length_features;
		
		double[] w = new double[length_features];
		double[] w_next = new double[length_features];
		
		for (int iter = 1; iter <= gd_iterations; iter++) {
			for (Instance instance : instances) {
				int yi = Integer.parseInt(instance.getLabel().toString());
				HashMap<Integer, Double> tmpmap = new HashMap<Integer, Double>();
				tmpmap = instance.getFeatureVector().getMap();
				double g = 0;
				for (int key : tmpmap.keySet()) {
					g += tmpmap.get(key) * w[key-1];
				}
				g = 1.0/(1 + Math.exp(-g));
				for (int key : tmpmap.keySet()) {
					double xij = tmpmap.get(key);
					w_next[key-1] += gd_eta*(yi-g)*xij;
				}
			}
			w = w_next.clone();
		}
		this._w = w;
//		for (int i = 0; i < length_features; i++) {
//			System.out.println(w[i]);
//		}
	}

	@Override
	public Label predict(Instance instance) {
		// TODO Auto-generated method stub
		HashMap<Integer, Double> map = new HashMap<Integer, Double>();
		map = instance.getFeatureVector().getMap();
		
		double prob = 0;
		for (int key : map.keySet()) {
			if (this._length_features >= key) { // Make sure the feature is learned in training.
				prob += map.get(key)*this._w[key-1];	
			}			
		}
		prob = 1.0/(1+Math.exp(-prob));
		
		Label label = null;
		if (prob >= 0.5) {
			label = new ClassificationLabel(1);
		} else {
			label = new ClassificationLabel(0);
		}
		return label;
	}

}
