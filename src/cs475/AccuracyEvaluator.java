package cs475;

import java.io.Serializable;
import java.util.List;

public class AccuracyEvaluator extends Evaluator implements Serializable {
	private static final long serialVersionUID = 1L;
		
	List<Instance> _instances;
	Predictor _predictor;
	
	public AccuracyEvaluator(List<Instance> instances, Predictor predictor) {
		this._instances = instances;
		this._predictor = predictor;
	}
	
	@Override
	public double evaluate(List<Instance> instances, Predictor predictor) {
		Label labelOrigin = null; // Origin label in the data.
		Label labelPredict = null; // Predicted label by the algorithm.
		int correct = 0; // Number of correct label predictions.
		int n = 0; // Total number of predictions.
		for (Instance instance : instances) {
			labelPredict = predictor.predict(instance);
			labelOrigin = instance.getLabel();
			if (labelOrigin != null) {
				n++;
				if (labelOrigin.toString().equalsIgnoreCase(labelPredict.toString())) {
					// If the predicted label equals the original label, then count plus 1.
					correct += 1;
				}
			}
		}
		if (n == 0) {
			return 0;
		} else {
			return correct*1.0/n;
		}
	}
}
