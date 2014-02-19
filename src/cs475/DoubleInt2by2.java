package cs475;

import java.io.Serializable;

public class DoubleInt2by2 implements Serializable {
	private static final long serialVersionUID = 1L;
	
	double _double_num;
	int[][] _count = new int[2][2];

	public DoubleInt2by2(double double_num, int[][] count) {
		this._double_num = double_num;
		this._count = count;
	}
	
	public double getDouble() {
		return this._double_num;
	}

	public void setDouble(double double_num) {
		this._double_num = double_num;
	}

	public int[][] getArray() {
		return this._count;
	}

	public void setArray(int[][] count) {
		this._count = count;
	}
}