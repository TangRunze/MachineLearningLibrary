package cs475;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class StochasticKMeansPredictor extends Predictor implements Serializable {
	private static final long serialVersionUID = 1L;
	
	List<Instance> _instances;
	int _clustering_training_iterations;
	ArrayList<HashMap<Integer, Double>> _mu;
	int _num_clusters;
	
	public StochasticKMeansPredictor(List<Instance> instances, Integer clustering_training_iterations, Integer num_clusters) {
		this._instances = instances;
		this._clustering_training_iterations = clustering_training_iterations;
		this._num_clusters = num_clusters;
		
		this._mu = new ArrayList<HashMap<Integer, Double>>(instances.size());
		for (int i = 1; i <= num_clusters; i++) {
			this._mu.add(instances.get(i-1).getFeatureVector().getMap());
		}
	}

	@Override
	public void train(List<Instance> instances) {
		// TODO Auto-generated method stub
		int clustering_training_iterations = this._clustering_training_iterations;
		ArrayList<HashMap<Integer, Double>> mu = this._mu;
		int num_clusters = this._num_clusters;
		int N = instances.size();
		int[] r = new int[N];
		int[] r_pre = new int[N];
		int[] count = new int[num_clusters];
		
		for (int iter = 1; iter <= clustering_training_iterations; iter++) {
			for (int i = 0; i < N; i++) {
				// E-step
				Instance instance = instances.get(i);
				HashMap<Integer, Double> map = instance.getFeatureVector().getMap();
				double min = Double.MAX_VALUE;
				int k = 0;
				for (int j = 1; j <= num_clusters; j++) {
					HashMap<Integer, Double> muj = mu.get(j-1);
					double total = 0;
					for (int key : map.keySet()) {
						double tmp = map.get(key);
						if (muj.containsKey(key)) {
							tmp = tmp - muj.get(key);
						}
						total = total + tmp*tmp;
					}
					for (int key : muj.keySet()) {
						if (!(map.containsKey(key))) {
							double tmp = muj.get(key);
							total = total + tmp*tmp;
						}
					}
					if (total < min) {
						min = total;
						k = j;
					}
				}
				r[i] = k;

				// M-step
				if (r[i] != r_pre[i]) {
					for (int key : mu.get(r[i]-1).keySet()) {
						double tmp = mu.get(r[i]-1).get(key)*count[r[i]-1];
						if (map.containsKey(key)) {
							tmp += map.get(key);
						}
						tmp = tmp/(count[r[i]-1]+1);
						mu.get(r[i]-1).put(key, tmp);
					}
					for (int key : map.keySet()) {
						if (!(mu.get(r[i]-1).containsKey(key))) {
							double tmp = map.get(key)/(count[r[i]-1]+1);
							mu.get(r[i]-1).put(key, tmp);
						}
					}
					count[r[i]-1]++;
					if (r_pre[i] != 0) {
						if (count[r_pre[i]-1] == 1) {
							HashMap<Integer, Double> tmpmap = new HashMap<Integer, Double>();
							mu.set(r_pre[i]-1, tmpmap);
						} else {
							for (int key : mu.get(r_pre[i]-1).keySet()) {
								double tmp = mu.get(r_pre[i]-1).get(key)*count[r_pre[i]-1];
								if (map.containsKey(key)) {
									tmp -= map.get(key);
								}
								tmp = tmp/(count[r_pre[i]-1]-1);
								mu.get(r_pre[i]-1).put(key, tmp);
							}
							for (int key : map.keySet()) {
								if (!(mu.get(r_pre[i]-1).containsKey(key))) {
									double tmp = - map.get(key)/(count[r_pre[i]-1]-1);
									mu.get(r_pre[i]-1).put(key, tmp);
								}
							}
						}
						count[r_pre[i]-1]--;
					}
				}
				r_pre[i] = r[i];
			}
		}
		this._mu = mu;
	}

	@Override
	public Label predict(Instance instance) {
		// TODO Auto-generated method stub
		ArrayList<HashMap<Integer, Double>> mu = this._mu;
		int num_clusters = this._num_clusters;

		HashMap<Integer, Double> map = instance.getFeatureVector().getMap();
		double min = Double.MAX_VALUE;
		int k = 0;
		
		for (int j = 1; j <= num_clusters; j++) {
			HashMap<Integer, Double> muj = mu.get(j-1);
			double total = 0;
			for (int key : map.keySet()) {
				double tmp = map.get(key);
				if (muj.containsKey(key)) {
					tmp = tmp - muj.get(key);
				}
				total += tmp*tmp;
			}
			for (int key : muj.keySet()) {
				if (!(map.containsKey(key))) {
					double tmp = muj.get(key);
					total += tmp*tmp;
				}
			}
			if (total < min) {
				min = total;
				k = j;
			}
			//System.out.print(" " + Double.toString(Math.sqrt(total)));
		}
		//System.out.println(" ");
		Label label = null;
		label = new ClassificationLabel(k-1);
		return label;
	}

}
