package cs475;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

public class FeatureSelection implements Serializable {
	private static final long serialVersionUID = 1L;
	
	List<Instance> _instances;
	
	public FeatureSelection(List<Instance> instances) {
		this._instances = instances;
		
	}

	public List<Instance> select(Integer num_features_to_select) {
		// Double is for the value of the feature; Int[2][2] is a prob table.
		HashMap<Integer, DoubleInt2by2> map = new HashMap<Integer, DoubleInt2by2>();
		HashMap<Integer, Double> entropymap = new HashMap<Integer, Double>();
		
		int num_instance = 0; // number of instance
		int[] label_count = new int[2]; 
		// Calculate the threshold of each feature (mean).
		for (Instance instance : this._instances) {
			num_instance++;
			HashMap<Integer, Double> tmpmap = instance.getFeatureVector().getMap();
			int labelint = Integer.parseInt(instance.getLabel().toString());
			label_count[labelint] += 1;
			for (int key : tmpmap.keySet()) {
				double value = tmpmap.get(key);
				int[][] count = new int[2][2];
				if (map.containsKey(key)) {
					value += map.get(key).getDouble();
				}
				DoubleInt2by2 doubleint2by2 = new DoubleInt2by2(value, count);		
				map.put(key, doubleint2by2);
			}
		}
		//int[][] emptyArray = new int[2][2];
		for (int key : map.keySet()) {
			double value = map.get(key).getDouble();
			int[][] count = map.get(key).getArray();
			DoubleInt2by2 doubleint2by2 = new DoubleInt2by2(value*1.0/num_instance, count);
			map.put(key, doubleint2by2);
		}
		
		for (Instance instance : this._instances) {
			HashMap<Integer, Double> tmpmap = instance.getFeatureVector().getMap();
			int labelint = Integer.parseInt(instance.getLabel().toString());
			for (int key : tmpmap.keySet()) {
				double value = tmpmap.get(key);
				int j = 0;
				if (value >= map.get(key).getDouble()) {
					j = 1;
				}
				double double_num = map.get(key).getDouble();
				int[][] count = new int[2][2];
				count = map.get(key).getArray().clone();
				count[labelint][j] += 1;
				DoubleInt2by2 doubleint2by2 = new DoubleInt2by2(double_num, count);
				map.put(key, doubleint2by2);
			}
		}
		
		for (int key : map.keySet()) {
			int[][] count = new int[2][2];
			count = map.get(key).getArray().clone();
			int j = 0;
			double double_num = map.get(key).getDouble();
			if (0 >= double_num) {
				j = 1;
			}
			count[0][j] = label_count[0] - count[0][1-j];
			count[1][j] = label_count[1] - count[1][1-j];
			DoubleInt2by2 doubleint2by2 = new DoubleInt2by2(double_num, count);
			map.put(key, doubleint2by2);
		}
		
		int[] feature_id = new int[num_features_to_select];
		double[] feature_entropy = new double[num_features_to_select];
		for (int key : map.keySet()) {
			// Calculate the entropy.
			int[][] count = map.get(key).getArray();
			int n = count[0][0]  + count[0][1] + count[1][0] + count[1][1];
			double entropy = 0;
			for (int i = 0; i <= 1; i++) {
				for (int j = 0; j<=1; j++) {
					if (count[i][j]*1.0/n != 0) {
						entropy += count[i][j]*1.0/n*Math.log(count[i][j]*1.0/(count[0][j]+count[1][j]));
					}
				}
			}
			
			// See if it is possibly one of the features with the largest entropy.
			int index = 0;
			while ((index < num_features_to_select) && (feature_id[index] != 0) && (feature_entropy[index] >= entropy)) {
				index++;
			}
			if (index < num_features_to_select) {
				if (feature_id[num_features_to_select - 1] != 0) {
					entropymap.remove(feature_id[num_features_to_select - 1]);
				}
				for (int i = num_features_to_select - 1; i > index; i--) {
					feature_id[i] = feature_id[i - 1];
					feature_entropy[i] = feature_entropy[i - 1];
				}
				feature_id[index] = key;
				feature_entropy[index] = entropy;
				entropymap.put(key, entropy);
			}
			// Now selected features are stored in entropymap.
		}
		
//		for (int i = 0; i < num_features_to_select; i++) {
//			System.out.println(feature_id[i] + " " + feature_entropy[i]);
//		}
		
		// Build a new data which only contains the selected features.
		List<Instance> instances_new = new ArrayList<Instance>();
		for (Instance instance : this._instances) {
			FeatureVector featurevector = instance.getFeatureVector();
			FeatureVector featurevector_new = new FeatureVector();
			HashMap<Integer, Double> tmpmap = featurevector.getMap();
			// Check if we should delete the instance, which means
			// all the features are not selected.
			int instance_add = 0; 
			for (int key : tmpmap.keySet()) {
				if (entropymap.containsKey(key)) {
					featurevector_new.add(key, tmpmap.get(key));
					instance_add = 1;
				}
			}
			// Add this instance to the new instances.
			if (instance_add == 1) {
				Label label = instance.getLabel();
				Instance instance_new = new Instance(featurevector_new, label);
				instances_new.add(instance_new);
			}
		}
		
		return instances_new;
	}
	
}