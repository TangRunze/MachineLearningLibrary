
package cs475.chainMRF;
import cs475.chainMRF.*;

public class SumProduct {

	private ChainMRFPotentials potentials;
	// add whatever data structures needed

	public SumProduct(ChainMRFPotentials p) {
		this.potentials = p;
	}

	public double[] marginalProbability(int x_i) {
		// TODO
		ChainMRFPotentials p = this.potentials;
		int n = p.chainLength();
		int k = p.numXValues();
		double[][] mu_fx = new double[2*n][k+1];
		double[][] mu_xf = new double[n+1][k+1];
		double[] marginal = new double[k+1];
		
		// Left to right
		for (int i = 1; i < x_i; i++) {
			for (int ind = 1; ind <= k; ind++) {
				mu_fx[i][ind] = p.potential(i, ind);
			}
			if (i == 1) {
				for (int ind = 1; ind <= k; ind++) {
					mu_xf[i][ind] = mu_fx[i][ind];
				}
			} else {
				for (int ind = 1; ind <= k; ind++) {
					mu_xf[i][ind] = mu_fx[i][ind]*mu_fx[n+i-1][ind];
				}
			}
			for (int ind1 = 1; ind1 <=k; ind1++) {
				mu_fx[n+i][ind1] = 0;
				for (int ind2 = 1; ind2 <=k; ind2++) {
					mu_fx[n+i][ind1] += p.potential(n+i, ind2, ind1)*mu_xf[i][ind2];
				}
			}
		}
		
		// Right to left
		for (int i = n; i > x_i; i--) {
			for (int ind = 1; ind <= k; ind++) {
				mu_fx[i][ind] = p.potential(i, ind);
			}
			if (i == n) {
				for (int ind = 1; ind <= k; ind++) {
					mu_xf[i][ind] = mu_fx[i][ind];
				}
			} else {
				for (int ind = 1; ind <= k; ind++) {
					mu_xf[i][ind] = mu_fx[i][ind]*mu_fx[n+i][ind];
				}
			}
			for (int ind1 = 1; ind1 <= k; ind1++) {
				mu_fx[n+i-1][ind1] = 0;
				for (int ind2 = 1; ind2 <=k; ind2++) {
					mu_fx[n+i-1][ind1] += p.potential(n+i-1, ind1, ind2)*mu_xf[i][ind2];
				}
			}
		}
		
		double total = 0;
		for (int ind = 1; ind <= k; ind++) {
			mu_fx[x_i][ind] = p.potential(x_i, ind);
		}
		if (x_i == 1) {
			for (int ind = 1; ind <= k; ind++) {
				marginal[ind] = mu_fx[1][ind]*mu_fx[n+1][ind];
				total += marginal[ind];
			}			
		} else if (x_i == n) {
			for (int ind = 1; ind <= k; ind++) {
				marginal[ind] = mu_fx[2*n-1][ind]*mu_fx[n][ind];
				total += marginal[ind];
			}
		} else {
			for (int ind = 1; ind <= k; ind++) {
				marginal[ind] = mu_fx[n+x_i-1][ind]*mu_fx[x_i][ind]*mu_fx[n+x_i][ind];
				total += marginal[ind];
			}
		}
		// Normalization
		for (int ind = 1; ind <= k; ind++) {
			marginal[ind] = marginal[ind]/total;
		}
		return marginal;
	}

}

