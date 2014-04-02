package cs475;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class LambdaMeansPredictor extends Predictor implements Serializable {
	private static final long serialVersionUID = 1L;
	
	List<Instance> _instances;
	int _clustering_training_iterations;
	ArrayList<HashMap<Integer, Double>> _mu;
	int _K = 0;
	double _cluster_lambda;
	HashMap<Integer, Double> _xmean_map;
	
	public LambdaMeansPredictor(List<Instance> instances, Integer clustering_training_iterations, Double cluster_lambda) {
		this._instances = instances;
		this._clustering_training_iterations = clustering_training_iterations;
		
		HashMap<Integer, Double> xmean_map = new HashMap<Integer, Double>();
		for (Instance instance : instances) {
			HashMap<Integer, Double> map_tmp = instance.getFeatureVector().getMap();
			for (int key : map_tmp.keySet()) {
				double tmp = map_tmp.get(key);
				if (xmean_map.containsKey(key)) {
					tmp += xmean_map.get(key);
				}
				xmean_map.put(key, tmp);
			}
		}
		
		int N = instances.size();
		for (int key : xmean_map.keySet()) {
			double tmp = xmean_map.get(key);
			tmp = tmp/N;
			xmean_map.put(key, tmp);
		}
		
		this._xmean_map = xmean_map;
		
		if (cluster_lambda == 0) {
			this._cluster_lambda = 0;
			for (Instance instance : instances) {
				double tmp1 = 0;
				HashMap<Integer, Double> map_tmp = instance.getFeatureVector().getMap();
				for (int key : xmean_map.keySet()) {
					double tmp = xmean_map.get(key);
					if (map_tmp.containsKey(key)) {
						tmp = tmp - map_tmp.get(key);
					}
					tmp1 += tmp*tmp;
				}
				this._cluster_lambda += Math.sqrt(tmp1);
			}
			this._cluster_lambda = this._cluster_lambda/N;
		} else {
			this._cluster_lambda = cluster_lambda;
		}
		this._mu = new ArrayList<HashMap<Integer, Double>>(instances.size());
		this._K += 1;
		this._mu.add(xmean_map);
	}

	@Override
	public void train(List<Instance> instances) {
		// TODO Auto-generated method stub
		int clustering_training_iterations = this._clustering_training_iterations;
		ArrayList<HashMap<Integer, Double>> mu = this._mu;
		int K = this._K;
		int N = instances.size();
		int[] r = new int[N];		
		
		for (int iter = 1; iter <= clustering_training_iterations; iter++) {
			// E-step
			for (int i = 0; i < N; i++) {
				Instance instance = instances.get(i);
				HashMap<Integer, Double> map = instance.getFeatureVector().getMap();
				double min = Double.MAX_VALUE;
				int k = 0;
				for (int j = 1; j <= K; j++) {
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
				min = Math.sqrt(min);
				if (min <= this._cluster_lambda) {
					r[i] = k;
				} else {
					K += 1;
					r[i] = K;
					mu.add(map);
				}
			}
			
			//System.out.println(Integer.toString(K));
			
			// M-step
			int[] count = new int[K];
			// Initialize mu_new
			ArrayList<HashMap<Integer, Double>> mu_new = new ArrayList<HashMap<Integer, Double>>(instances.size());
			for (int j = 1; j <= K; j++) {
				HashMap<Integer, Double> map = new HashMap<Integer, Double>();
				mu_new.add(map);
			}
			for (int i = 0; i < N; i++) {
				Instance instance = instances.get(i);
				HashMap<Integer, Double> map = instance.getFeatureVector().getMap();
				int k = r[i];
				count[k-1] += 1;
				for (int key : map.keySet()) {
					double tmp = map.get(key);
					if (mu_new.get(k-1).containsKey(key)) {
						tmp += mu_new.get(k-1).get(key);
					}
					mu_new.get(k-1).put(key, tmp);
				}
			}
			for (int j = 1; j <= K; j++) {
				if (count[j-1]!=0) {
					for (int key : mu_new.get(j-1).keySet()) {
						mu_new.get(j-1).put(key, mu_new.get(j-1).get(key)/count[j-1]);
					}
				}
			}
			mu = mu_new;
		}
		this._mu = mu;
		this._K = K;
	}

	@Override
	public Label predict(Instance instance) {
		// TODO Auto-generated method stub
		ArrayList<HashMap<Integer, Double>> mu = this._mu;
		int K = this._K;

		HashMap<Integer, Double> map = instance.getFeatureVector().getMap();
		double min = Double.MAX_VALUE;
		int k = 0;
		
		for (int j = 1; j <= K; j++) {
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
		}
		
		Label label = null;
		label = new ClassificationLabel(k-1);
		return label;
	}

}
