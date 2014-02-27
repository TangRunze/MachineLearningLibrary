package cs475;

import java.util.HashMap;

public abstract class KernelFun {

	public abstract double calculateKernel(HashMap<Integer, Double> map1, HashMap<Integer, Double> map2, int polynomial_kernel_exponent);
}
