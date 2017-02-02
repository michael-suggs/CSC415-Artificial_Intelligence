package artmap;

import java.util.*;


public class Dartmap implements Learner {
	public static final int FUZZY   = 0;
	public static final int DEFAULT = 1;
	public static final int IC      = 2;
	public static final int DISTRIB = 3;

	protected int NetworkType;    // Controls algorithm used

	protected int     M;          // Number of inputs (pre-CC)
	protected int     L;	      // Number of output classes (1..L, not 0..(L-1))

	protected float   RhoBar;     // Baseline vigilance
	protected float   RhoBarTest; // Baseline vigilance
	protected float   Alpha;      // Signal rule parameter
	protected float   Beta;       // Learning rate
	protected float   Eps;        // Match tracking parameter
	protected float   P;          // CAM rule power parameter

	public void setNetworkType (int   v) { NetworkType = v; }
	public void setM           (int   v) { M	       = v; }
	public void setL           (int   v) { L	       = v; }
	public void setRhoBar      (float v) { RhoBar      = v; }
	public void setRhoBarTest  (float v) { RhoBarTest  = v; }
	public void setAlpha       (float v) { Alpha       = v; }
	public void setBeta        (float v) { Beta        = v; }
	public void setEps         (float v) { Eps         = v; }
	public void setP           (float v) { P           = v; }
	
	public int getNetworkType()         { return NetworkType; }
	public int 	    getM()              { return M;           }
	public int 	    getL()              { return L;           }
	public float       getRhoBar()      { return RhoBar;      }
	public float       getRhoBarTest()  { return RhoBarTest;  }
	public float       getAlpha()       { return Alpha;       }
	public float       getBeta()        { return Beta;        }
	public float       getEps()         { return Eps;         }
	public float       getP()           { return P;           }
	Random randGen = new Random(1);
	
	private static final float      tinyNum              = 0.0000001f;
	public static final float       default_RhoBar       =  0.0f;
	public static final float       default_RhoBarTest   =  0.0f;
	public static final float       default_Alpha        =  0.01f;
	public static final float       default_Beta         =  1.0f;
	public static final float       default_Eps          = -0.001f;
	public static final float       default_P            =  1.0f;

	public static final int default_NetworkType  = DEFAULT;

	public static final int         default_initNumNodes = 20000;
	public static final float       default_F2growthRate = 2.0f;
	
	int     C;      // Number of committed nodes
	
	int     J;      // In WTA mode, index of the winning node
	int     K;      // The target class (1..L, not 0..(L-1))

	float   rho;    // Current vigilance

	// Indices - i: 1..M, j: 1..C; k: 1..L
	float A[];          // i - Complement-coded input
	float x[];          // i - F1, matching
	float y[];          // j - F2, coding
	float Y[];          // j - F3, counting
	float T[];          // j - Total F0->F2
	float S[];          // j - Phasic F0->F2
	float H[];          // j - Tonic F0->F2 (Capital Theta)
	float c[];          // j - F2->F3
	boolean  lambda[];  // j - T if node is eligible, F otherwise
	float sigma_i[];    // i - F3->F1
	float sigma_k[];    // k - F3->F0ab
	int   kap[];        // j - F3->Fab (small kappa)
	float tIj[];        // ij - F0->F2 (tau sub ij)
	float tJi[];        // ji - F3->F1 (tau sub ji)

	// Utility variables (not in algorithm)
	float  Tu;              // Uncommitted node activation
	float  sum_x;           // To avoid recomputing norm
	int    _2M;             // To keep from calculating 2*M over and over
	int    N;               // Growable upper bound on coding nodes

	/**
	 * @param M
	 * @param L
	 */
	public Dartmap (int M, int L) {
		_2M = 2*M;
	
		setM (M);
		setL (L);
	
		N           = default_initNumNodes;
		NetworkType = default_NetworkType;
		RhoBar      = default_RhoBar;
		RhoBarTest  = default_RhoBarTest;
		Alpha       = default_Alpha;
		Beta        = default_Beta;
		Eps         = default_Eps;
		P           = default_P;
	
		A = new float[_2M];
		x = new float[_2M];
		sigma_i = new float[_2M];
		sigma_k = new float[L];
	
		y = new float[N];
		Y = new float[N];
		T = new float[N];
		S = new float[N];
		H = new float[N];
		c = new float[N];
		lambda = new boolean[N];
		kap = new int[N];
	
		tIj = new float[_2M*N];
		tJi = new float[N*_2M];
	
		for (int j = 0; j < N; j++) {
			c[j] = 0.0f;
			for (int i = 0; i < _2M; i++) { tIj[j*_2M + i] = tJi[j*_2M + i] = 0.0f; }
		}
	
		C = 0;

		Tu = (float) M;
	}
	
	public void clear() {
		C = 0;
	}
	
	/**
	 * 
	 */
	public void train (float  a[], int _K) {
		boolean d = (NetworkType == DISTRIB) ? true : false;
		boolean needNewNode = false;

		K = _K;                    // Save target class 
		
		complementCode (a);
		
		if (C == 0) {              // New network - commit a new node
			needNewNode = true;
		} else {
			rho = RhoBar;          // Reset network vigilance to baseline

			int eligibleNodes = F0_to_F2_signal();

			for (;;) {                       // Outer loop: Match tracking 
				boolean passedVigilance = false;

				while (eligibleNodes > 0) {    // Inner loop: Vigilance 
					if (d) {
						CAM_distrib(); F1signal_distrib();
					} else { 
						CAM_WTA();     F1signal_WTA();  eligibleNodes--; 
					}

					if (passesVigilance()) { passedVigilance = true; break; }

					d = false;
				} // Failed vigilance, try again

				if (passedVigilance == false) { // Fell through without finding candidate
					needNewNode = true; break; 
				}

				// Passed vigilance criterion
				int Kprime = (d ? prediction_distrib() : prediction_WTA());

				if (Kprime == K) break;

				matchTracking();

				d = false;
			} // predicted wrong class, try again
		}

		if (needNewNode) {
			newNode();
		} else {
			if (NetworkType == DISTRIB) { creditAssignment(); resonance_distrib(); } else { resonance_WTA();  }
		}	
	}
	
	/**
	 * 
	 */
	public float[] test (float a[]) {
		boolean dontKnow = false;
		boolean d = (NetworkType == FUZZY) ? false : true;

		rho = RhoBarTest;          

		complementCode(a);  

		int eligibleNodes = F0_to_F2_signal();

		if (eligibleNodes > 0) {
			if (d) {
				CAM_distrib(); F1signal_distrib();
			} else { 
				CAM_WTA();     F1signal_WTA();
			}
			if (passesVigilance()) 
				if (d) {prediction_distrib(); } else { prediction_WTA(); }
			else {
				dontKnow = true;
			}
		} else {
			dontKnow = true;
		}

		if (dontKnow) {
			for (int k = 0; k < L; k++) { sigma_k[k] = 1; } // "Don't know" sets all outputs to 1
		}		
		
		return sigma_k;
	}
		
	/**
	 * @param a
	 */
	private void complementCode(float  a[]) {
		for (int i = 0; i < M; i++) {
			A[i]   =       a[i];
			A[i+M] = 1.0f - a[i];
		}
	}
		
	/**
	 * @return
	 */
	private int  F0_to_F2_signal() {
		int eligibleNodes = 0;
	
		for (int j = 0; j <   C; j++)  {
			S[j] = H[j] = 0.0f;
	
			for (int i = 0; i < _2M; i++) {
				S[j] += Math.min (A[i], (1 - tauIj(i, j)));
				H[j] += tauIj(i, j);
			}
	
			T[j] = (S[j] + (1 - Alpha) * H[j]);
	
			if (T[j] >= Tu) { eligibleNodes++; lambda[j] = true; } else lambda[j] = false;
		}
		
		return eligibleNodes;	
	}
		
	/**
	 * 
	 */
	private void newNode() {
		if (C == (N-1)) System.out.println ("Out of F2 nodes!");
	
		J = C++;
	
		kap[J] = K;
	
		for (int k = 0; k < L; k++) { sigma_k[k] = 0; } 
		sigma_k[K] = 1;
	
		for (int j = 0; j < C; j++)  { y[j] = Y[j] = 0; }
		y[J] = Y[J] = 1;
	
		// Set initial values for the node's weights
		for (int i = 0; i < _2M; i++) { tIj[J*_2M + i] = tJi[J*_2M + i] = 1.0f - A[i]; }
		
		c[J] = 1; // Initialize instance count	
	}
	
	/**
	 * 
	 */
	private void CAM_distrib() {
		float   costArray[]  = new float[C];
		boolean ptBoxArray[] = new boolean [C];
		int numPtBoxes = 0;
	
		for (int j = 0; j < C; j++) { 
			if (lambda[j] && ((costArray[j] = cost(T[j])) < tinyNum)) {
				ptBoxArray[j] = true;
				numPtBoxes++; 
			} else {
				ptBoxArray[j] = false;
			}
		}
	
		if (numPtBoxes == 0) {
			float  sumOfInvCosts = 0.0f; // First compute denominator term 
			for (int j = 0; j < C; j++) { if (lambda[j]) sumOfInvCosts += 1.0f / (Math.pow (cost (T[j]), P)); }
			for (int j = 0; j < C; j++) { y[j] = ((lambda[j]) ? (1.0f / (float)(Math.pow (cost(T[j]), P) * sumOfInvCosts)) : 0.0f); }
		} else {
			for (int j = 0; j < C; j++) { y[j] = ((ptBoxArray[j]) ? (1.0f / numPtBoxes) : 0.0f); }    
		}
	
		// Calculate F3 activation:
		float  denom = 0.0f;
		if (NetworkType == DEFAULT) {
			for (int j = 0; j < C; j++) { denom += y[j]; }
			for (int j = 0; j < C; j++) { Y[j] = y[j] / denom; }
		} else {
			for (int j = 0; j < C; j++) { denom += c[j] * y[j]; }
			for (int j = 0; j < C; j++) { Y[j] = c[j] * y[j] / denom; }
		}
	}
	
	/**
	 * 
	 */
	private void CAM_WTA() {
		Vector<Integer> tied = new Vector<Integer>();
		float  maxTj = -1.0f;
		for (int j = 0; j < C; j++) { // Looking only at eligible nodes, find all maximal T[j]s
			if (lambda[j]) {
				if (T[j] > maxTj) { 
					tied.clear() ; tied.add(new Integer(j));
					maxTj = T[j];
				} else if (Math.abs (T[j]-maxTj) < tinyNum) {  // floating pt '==' 
					tied.add(new Integer(j));
				}
			}
		
			y[j] = Y[j] = 0;
		}
		
		// If tie for winner, choose at random from list of ties
		int idx = (tied.size() > 1) ? (randGen.nextInt(tied.size())) : 0;
		J = ((Integer)(tied.elementAt(idx))).intValue();
		
		y[J] = Y[J] = 1;
		
		lambda[J] = false;  // Node J no longer eligible
	}
	
	/**
	 * 
	 */
	private void F1signal_WTA(){
		// F3->F1 signal
		for (int i = 0; i < _2M; i++) {
			sigma_i[i] = (1 - tauJi (i, J));
		}
	
	}
	
	/**
	 * 
	 */
	private void F1signal_distrib() {
		// Calculate F3->F1 signal:
		for (int i = 0; i < _2M; i++) {
			sigma_i[i] = 0.0f;
		for (int j = 0; j < C; j++) {
			sigma_i[i] += pos (Y[j] - tauJi(i, j));
		}
		}
	}
	
	/**
	 * @return
	 */
	private boolean passesVigilance() {
		sum_x = 0.0f;
	
		for (int i = 0; i < _2M; i++) { sum_x += Math.min (A[i], sigma_i[i]);  }
	
		// Decrease rho by tiny amount - compensate for float inaccuracy
		return ((sum_x / M) >= (rho - tinyNum));
	}
	
	/**
	 * @return
	 */
	private int  prediction_distrib() {
		for (int k = 0; k < L; k++) { sigma_k[k] = 0.0f; }
		for (int j = 0; j < C; j++) { sigma_k[(int) kap[j]] += Y[j]; }
		for (int k = 0; k < L; k++) { // Handle float inaccuracies causing overflow
			if (sigma_k[k] > 1.0f) sigma_k[k] = 1.0f;
		}
		float  sigma_kp = -1.0f;
		int    Kprime   = -1;
	
		for (int k = 0; k < L; k++) { if (sigma_k[k] > sigma_kp) { Kprime = k; sigma_kp = sigma_k[k]; } }
	
		return Kprime;
	}
	
	/**
	 * @return
	 */
	private int  prediction_WTA() {
		for (int k = 0; k < L; k++) { sigma_k[k] = 0.0f; }
	
		int Kprime = -1;	
		Kprime = kap[J];
		sigma_k[Kprime]= 1.0f;
	
		return Kprime;
	}
	
	/**
	 * 
	 */
	private void matchTracking() {
		rho = (sum_x / M) + Eps;
	}
	
	/**
	 * 
	 */
	private void creditAssignment() {
		float  sum_y  = 0.0f;
		float  sum_cy = 0.0f;
	
		// F2 blackout
		for (int j = 0; j < C; j++) {
		if (kap[j] != K) { y[j] = 0.0f; } 
			else { sum_y  += y[j];   }
		}
	
		for (int j = 0; j < C; j++) {
		y[j] /= sum_y;                 // F2 activation
			sum_cy += c[j] * y[j];
		}
	
		for (int j = 0; j < C; j++) {
		Y[j] = c[j] * y[j] / sum_cy;   // F3 activation
		}
	
		// F3->F1 signal
		for (int i = 0; i < _2M; i++) {
		sigma_i[i] = 0.0f;
		for (int j = 0; j < C; j++) { sigma_i[i] += pos(Y[j] - tauJi(i, j)); }
		}
	}
	
	/**
	 * 
	 */
	private void resonance_distrib() {
		for (int j = 0; j < C; j++) {
			for (int i = 0; i < _2M; i++) {
				// Increase F0->F2 threshold (distributed instar)
				tIj[j*_2M + i] += Beta * pos(y[j] - tauIj(i, j) - A[i]);
	
				// Increase F3->F1 threshold (distributed outstar)
				if (sigma_i[i] != 0.0) {
					tJi[j*_2M + i] += Beta * (pos(sigma_i[i] - A[i]) / sigma_i[i]) * pos(Y[j] - tauJi(i, j));
				}
			}
	
			c[j] += y[j]; // Increase F2->F3 instance counting weights
		}
	}
	
	private void resonance_WTA() {
		// Use winner-take-all version of learning law
		for (int i = 0; i < _2M; i++) { tIj[J*_2M + i] = tJi[J*_2M + i] = tauIj(i,J) + Beta * pos (1.0f - tauIj(i,J) - A[i]); }
		
		c[J] += y[J]; // Increase F2->F3 instance counting weights
	}
	
	
	private float cost(float x) { 
		return ((2-Alpha)*M - x); 
	}
	
	public float  getOutput (int k) { return sigma_k[k]; }
	
	public int getMaxOutputIndex () { 
		  float maxEltVal      = Float.NEGATIVE_INFINITY;
		  int maxEltIdx        = -1;
		  boolean haveEntries  = false;
		  boolean allZeros     = true;
		  int numWinners       = 0;

		  for (int i = 0; i < L; i++) {
			if (sigma_k[i] != 0) allZeros = false;
			haveEntries = true;
		    if (sigma_k[i] > maxEltVal) {
		      maxEltVal = sigma_k[i];
		      maxEltIdx = i;
			  numWinners = 1;
		    } else if (Math.abs(sigma_k[i] - maxEltVal) < tinyNum) {
			  numWinners++;
			}
		  }

		  if (!haveEntries)    return -3;
		  if (allZeros)        return -2;
		  
		  if (numWinners > 1) {
		    for (int i = 0; i < L; i++) {
		      sigma_k[i] = (Math.abs(sigma_k[i] - maxEltVal) < tinyNum) ? 1.0f / ((float)numWinners) : 0.0f;
		    }
		    return -1;
		  } else {
		    for (int i = 0; i < L; i++) {
		      sigma_k[i] = 0.0f;
		    }
		    sigma_k[maxEltIdx] = 1.0f;
		    return maxEltIdx;
		  }
	}
	 
	public int         getC()            { return C;           }
	public int         getNodeClass (int j) { if ((j < 0) || (j > C)) { return -1; } else { return kap[j]; } }
	public int         getLtmRequired () { return C * M * 2 * 16; } // 16 = sizeof (float) in Java
		
	public float tauIj (int i, int j) {
		return tIj[j*_2M + i]; 
	}
	public float tauJi (int i, int j) {
		return tJi[j*_2M + i]; 
	}
	
	private float pos(float v) { 
		return (v > 0.0f) ? v : 0.0f; 
	}
		
	public void setParam (String name, String value) throws Exception {
		float val;
				
		if (name == "Model") {
			if (value == "fuzzy")   	 setNetworkType (Dartmap.FUZZY);
			else if (value == "default") setNetworkType (Dartmap.DEFAULT);
			else if (value == "ic")      setNetworkType (Dartmap.IC);
			else if (value == "distrib") setNetworkType (Dartmap.DISTRIB);
			else throw new Exception ("ArtmapParams::setValue() - Unknown Artmap model requested (" + value + ")");
			
		} else if (name == "RhoBar") {
			try {
				val = Float.parseFloat(value);
			} catch (NumberFormatException e) {
				throw new Exception ("ArtmapParams::setValue() - Couldn't read value of RhoBar"); 
			}
			if ((val < 0) || (val > 1.0)) throw new Exception ("ArtmapParams::setValue() - Rhobar out of range [0, 1]");
			RhoBar = val;
			
		} else if (name == "RhoBarTest") {
			try {
				val = Float.parseFloat(value);
			} catch (NumberFormatException e) {
				throw new Exception ("ArtmapParams::setValue() - Couldn't read value of RhoBarTest"); 
			}
			if ((val < 0) || (val > 1.0)) throw new Exception ("ArtmapParams::setValue() - RhobarTest out of range [0, 1]");
			RhoBarTest = val;
			
		} else if (name == "Alpha") {
			try {
				val = Float.parseFloat(value);
			} catch (NumberFormatException e) {
				throw new Exception ("ArtmapParams::setValue() - Couldn't read value of Alpha"); 
			}
			Alpha = val;
	
		} else if (name == "Beta") {
			try {
				val = Float.parseFloat(value);
			} catch (NumberFormatException e) {
				throw new Exception ("ArtmapParams::setValue() - Couldn't read value of Beta"); 
			}
			Beta = val;
			
		} else if (name == "Eps") {
			try {
				val = Float.parseFloat(value);
			} catch (NumberFormatException e) {
				throw new Exception ("ArtmapParams::setValue() - Couldn't read value of Eps"); 
			}
			Eps = val;
			
		} else if (name == "P") {
			try {
				val = Float.parseFloat(value);
			} catch (NumberFormatException e) {
				throw new Exception ("ArtmapParams::setValue() - Couldn't read value of P"); 
			}
			P = val;
		} else throw new Exception ("artmap::setParam() - Unknown param (" + name + ")");	
	}
}
