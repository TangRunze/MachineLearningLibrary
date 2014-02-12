package cs475;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;

public class EvenOddClassifier extends Predictor implements Serializable {
	private static final long serialVersionUID = 1L;
	
	List<Instance> _instances;
	Label _label = null;
	
	public EvenOddClassifier(List<Instance> instances) {
		this._instances = instances;
	}

//	public Label getLabel() {
//		return this._label;
//	}
	
	@Override
	public void train(List<Instance> instances) {
	}
	
	@Override
	public Label predict(Instance instance) {
		
		double EvenSum = 0.0;
		double OddSum = 0.0;
		Label label = null;
		
		// Build a HashMap counting the number of labels in the test data.
		FeatureVector featurevector = instance.getFeatureVector();
		HashMap<Integer, Double> map = featurevector.getMap();
		for (int key : map.keySet()) {
			if ((key % 2) == 0) {
				EvenSum += map.get(key);
			} else {
				OddSum += map.get(key);
			}
		}
		
		// If even-sum >= odd-sum, predict 1; Otherwise predict 0.
		if (EvenSum >= OddSum) {
			label = new ClassificationLabel(1);
		} else {
			label = new ClassificationLabel(0);
		}
		
		return label;
	}
}
