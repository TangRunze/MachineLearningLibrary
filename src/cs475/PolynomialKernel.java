package cs475;

import java.io.Serializable;
import java.util.HashMap;

public class PolynomialKernel extends KernelFun implements Serializable {
	private static final long serialVersionUID = 1L;
	
	//public LinearKernel() {
	//}
	
	@Override
	public double calculateKernel(HashMap<Integer, Double> map1, HashMap<Integer, Double> map2, int polynomial_kernel_exponent) {
		double total = 0;
		for (int key : map2.keySet()) {
			if (map1.containsKey(key)) {
				total += map1.get(key)*map2.get(key);
			}
		}
		return Math.pow((1 + total), polynomial_kernel_exponent);
	}
}
