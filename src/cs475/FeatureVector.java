package cs475;

import java.io.Serializable;

import java.util.*;

public class FeatureVector implements Serializable {

	HashMap<Integer, Double> map = new HashMap<Integer, Double>();
	
	public void add(int index, double value) {
		// TODO Auto-generated method stub
		map.put(index, value);
	}
	
	public double get(int index) {
		// TODO Auto-generated method stub
		return map.get(index);
	}
	
	public HashMap<Integer, Double> getMap() {
		return map;
	}

}
