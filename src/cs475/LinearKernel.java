package cs475;

import java.io.Serializable;
import java.util.HashMap;

public class LinearKernel extends KernelFun implements Serializable {
	private static final long serialVersionUID = 1L;
	
	//public LinearKernel() {
	//}
	
	@Override
	public double calculateKernel(HashMap<Integer, Double> map1, HashMap<Integer, Double> map2, int polynomial_kernel_exponent) {
		double total = 0;
		for (int key : map2.keySet()) {
			Double value = map1.get(key);
			if (value != null) {
				total += value*map2.get(key);
			}
		}
		return total;
	}
}
