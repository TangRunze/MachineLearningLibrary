package cs475;

import java.io.Serializable;

public class RegressionLabel extends Label implements Serializable {

	double _label;
	
	public RegressionLabel(double label) {
		// TODO Auto-generated constructor stub
		this._label = label;
	}

	@Override
	public String toString() {
		// TODO Auto-generated method stub
		//return String.valueOf(this._label);
		return Double.toString(this._label);
	}

}
