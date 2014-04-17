package cs475.chainMRF;
import cs475.chainMRF.*;

public class MaxSum {

	private ChainMRFPotentials potentials;
	private int[] assignments;
	// add whatever data structures needed

	public MaxSum(ChainMRFPotentials p) {
		this.potentials = p;
		assignments = new int[p.chainLength()+1];
	}
	
	public int[] getAssignments() {
		return this.assignments;
	}

	public double maxProbability(int x_i) {
		// TODO
		ChainMRFPotentials p = this.potentials;
		int n = p.chainLength();
		int k = p.numXValues();
		double[][] mu_fx = new double[2*n][k+1];
		double[][] mu_xf = new double[n+1][k+1];
		int[][] x_assignment = new int[n+1][k+1];
		int[] assignments = new int[p.chainLength()+1];
		
		double[][] mu_fx1 = new double[2*n][k+1];
		double[][] mu_xf1 = new double[n+1][k+1];
		double[] marginal = new double[k+1];
		
		double logp = Double.NEGATIVE_INFINITY;
		
		// Left to right
		for (int i = 1; i < x_i; i++) {
			for (int ind = 1; ind <= k; ind++) {
				mu_fx[i][ind] = Math.log(p.potential(i, ind));
				mu_fx1[i][ind] = p.potential(i, ind);
			}
			if (i == 1) {
				for (int ind = 1; ind <= k; ind++) {
					mu_xf[i][ind] = mu_fx[i][ind];
					mu_xf1[i][ind] = mu_fx1[i][ind];
				}
			} else {
				for (int ind = 1; ind <= k; ind++) {
					mu_xf[i][ind] = mu_fx[i][ind] + mu_fx[n+i-1][ind];
					mu_xf1[i][ind] = mu_fx1[i][ind]*mu_fx1[n+i-1][ind];
				}
			}
			for (int ind1 = 1; ind1 <=k; ind1++) {
				mu_fx[n+i][ind1] = Double.NEGATIVE_INFINITY;
				mu_fx1[n+i][ind1] = 0;
				for (int ind2 = 1; ind2 <=k; ind2++) {
					double tmp = Math.log(p.potential(n+i, ind2, ind1)) + mu_xf[i][ind2];
					if (tmp > mu_fx[n+i][ind1]) {
						mu_fx[n+i][ind1] = tmp;
						x_assignment[i][ind1] = ind2;
					}
					mu_fx1[n+i][ind1] += p.potential(n+i, ind2, ind1)*mu_xf1[i][ind2];
				}
			}
		}
		
		// Right to left
		for (int i = n; i > x_i; i--) {
			for (int ind = 1; ind <= k; ind++) {
				mu_fx[i][ind] = Math.log(p.potential(i, ind));
				mu_fx1[i][ind] = p.potential(i, ind);
			}
			if (i == n) {
				for (int ind = 1; ind <= k; ind++) {
					mu_xf[i][ind] = mu_fx[i][ind];
					mu_xf1[i][ind] = mu_fx1[i][ind];
				}
			} else {
				for (int ind = 1; ind <= k; ind++) {
					mu_xf[i][ind] = mu_fx[i][ind] + mu_fx[n+i][ind];
					mu_xf1[i][ind] = mu_fx1[i][ind]*mu_fx1[n+i][ind];
				}
			}
			for (int ind1 = 1; ind1 <= k; ind1++) {
				mu_fx[n+i-1][ind1] = Double.NEGATIVE_INFINITY;
				mu_fx1[n+i-1][ind1] = 0;
				for (int ind2 = 1; ind2 <=k; ind2++) {
					double tmp = Math.log(p.potential(n+i-1, ind1, ind2)) + mu_xf[i][ind2];
					if (tmp > mu_fx[n+i-1][ind1]) {
						mu_fx[n+i-1][ind1] = tmp;
						x_assignment[i][ind1] = ind2;
					}
					mu_fx1[n+i-1][ind1] += p.potential(n+i-1, ind1, ind2)*mu_xf1[i][ind2];
				}
			}
		}
		
		double total = 0;
		for (int ind = 1; ind <= k; ind++) {
			mu_fx[x_i][ind] = Math.log(p.potential(x_i, ind));
			mu_fx1[x_i][ind] = p.potential(x_i, ind);
		}
		if (x_i == 1) {
			for (int ind = 1; ind <= k; ind++) {
				double tmp = mu_fx[1][ind] + mu_fx[n+1][ind];
				if (tmp > logp) {
					logp = tmp;
					assignments[x_i] = ind;
				}
				marginal[ind] = mu_fx1[1][ind]*mu_fx1[n+1][ind];
				total += marginal[ind];
			}			
		} else if (x_i == n) {
			for (int ind = 1; ind <= k; ind++) {
				double tmp = mu_fx[2*n-1][ind] + mu_fx[n][ind];
				if (tmp > logp) {
					logp = tmp;
					assignments[x_i] = ind;
				}
				marginal[ind] = mu_fx1[2*n-1][ind]*mu_fx1[n][ind];
				total += marginal[ind];
			}
		} else {
			for (int ind = 1; ind <= k; ind++) {
				double tmp = mu_fx[n+x_i-1][ind] + mu_fx[x_i][ind] + mu_fx[n+x_i][ind];
				if (tmp > logp) {
					logp = tmp;
					assignments[x_i] = ind;
				}
				marginal[ind] = mu_fx1[n+x_i-1][ind]*mu_fx1[x_i][ind]*mu_fx1[n+x_i][ind];
				total += marginal[ind];
			}
		}
		// Normalization
		logp = logp - Math.log(total);
		
		// Assignments
		if (x_i == 1) {
			for (int ind = 2; ind <= n; ind++) {
				assignments[ind] = x_assignment[ind][assignments[ind-1]];
			}
		} else if (x_i == n) {
			for (int ind = n - 1; ind >= 1; ind--) {
				assignments[ind] = x_assignment[ind][assignments[ind+1]];
			}
		} else {
			for (int ind = x_i - 1; ind >= 1; ind--) {
				assignments[ind] = x_assignment[ind][assignments[ind+1]];
			}
			for (int ind = x_i + 1; ind <= n; ind++) {
				assignments[ind] = x_assignment[ind][assignments[ind-1]];
			}
		}
		
		this.assignments = assignments;
		return logp;
	}
}
