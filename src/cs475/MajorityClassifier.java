package cs475;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;

public class MajorityClassifier extends Predictor implements Serializable {
	private static final long serialVersionUID = 1L;
	
	List<Instance> _instances;
	Label _label = null;
	
	public MajorityClassifier(List<Instance> instances) {
		this._instances = instances;
	}

//	public Label getLabel() {
//		return this._label;
//	}
	
	@Override
	public void train(List<Instance> instances) {
		// LabelInt is a new class which contains Label & int.
		HashMap<String, LabelInt> map = new HashMap<String, LabelInt>();
		
		int max = 0;
		
		// Build a HashMap counting the number of labels in the training data.
		for (Instance instance : instances) {
			LabelInt labelint = new LabelInt(null, 0);
			
			Label _labeltmp = instance.getLabel();
			String key = _labeltmp.toString();
			
			// If key is not in the HashMap, add it. Otherwise get the frequency of the key and add 1.
			if (map.get(key) == null) {
				labelint.setLabel(_labeltmp);
				labelint.setValue(1);
			}
			else {
				labelint = map.get(key);
				int value = labelint.getValue() + 1;
				labelint.setValue(value);
			}
			map.put(key, labelint);
		}
		// Find the most common label.
		for (String key : map.keySet()) {
			LabelInt labelint = new LabelInt(null, 0);
			
			labelint = map.get(key);
			if (labelint.getValue() > max) {
				max = labelint.getValue();
				this._label = labelint.getLabel();
			}
			// When two labels are tied, pick one at random.
			else if ((labelint.getValue() == max) && (Math.random() > 0.5)) {
				max = labelint.getValue();
				this._label = labelint.getLabel();
			}
		}
	}
	
	@Override
	public Label predict(Instance instance) {
		return this._label;
	}
}
