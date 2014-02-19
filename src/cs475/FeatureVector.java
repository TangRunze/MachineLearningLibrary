package cs475;

import java.io.Serializable;
import java.util.*;

public class FeatureVector implements Serializable {

	HashMap<Integer, Double> _map = new HashMap<Integer, Double>();
	int _MaxID = 0;
	
	public void add(int index, double value) {
		// TODO Auto-generated method stub
		this._map.put(index, value);
	}
	
	public void remove(int index) {
		this._map.remove(index);
	}
	
	public double get(int index) {
		// TODO Auto-generated method stub
		return this._map.get(index);
	}
	
	public HashMap<Integer, Double> getMap() {
		return this._map;
	}
	
	public int getMaxID() {
		for (int key : this._map.keySet()) {
			if (key > this._MaxID) {
				this._MaxID = key;
			}			
		}
		return this._MaxID;
	}
	
	public int checkKey(int index) {
		if (!this._map.containsKey(index)) {
			return 0;
		} else {
			return 1;
		}
	}

}
