/*******************************************************************************
 * artmap.cpp, v.1 (7/5/01)
 *
 * Description:
 *	Implementation of Fuzzy ARTMAP, ART-EMAP, ARTMAP-IC, and dARTMAP (v.2,
 *	the 1997 version) along with sample training/testing set for simple
 *	classifcation only (no ARTb module).
 *	See README and http://www.cns.bu.edu/~artmap for further details.
 * Compilation (in Unix):
 *	gcc artmap.cpp -o artmap -lm
 * Authors:
 *	Suhas Chelian, Norbert Kopco
 ******************************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/*******************************************************************************
 * Constants and Parameters
 ******************************************************************************/
// ARTMAP type.  Toggle EXACTLY one flag.
#define FUZZY_ARTMAP	1
#define ARTMAP_IC	0
#define ART_EMAP	0
#define DIST_ARTMAP	0

// Constants definitions and training/testing Data
#define M   2		// Dimensionality of input vectors (not including complement
			//	coding)
#define L   2		// Number of output classes
#define EPOCHS 1	// Number of training epochs
#define MAX_F2_SIZE 100	// Max number of F2 nodes.  Increase this if you run out
			//	in training.

#define TRAIN_N 8	// Number of training points
double	input[TRAIN_N][M] = {{.8,.5},{.5,.2},{.8,.8},{.7,.1},{1,1},{1,1},{.6,.4},{.2,.3}};
int	output[TRAIN_N]   = { 1, 1, 0, 0, 0, 0, 1, 1 };

#define TEST_N 8	// Number of testing points
double	te_input[TEST_N][M] = {{.2,.9},{.9,.6},{.6,.6},{.9,.8},{.7,.5},{.2,.7},{.4,.9},{.9,.7}};
int te_output[TEST_N] = { 1, 1, 0, 1, 0, 1, 1, 1};

// Parameters
double alpha =		.01;	// CBD and Weber law signal parameter
double p =		1.;    	// CAM rule power
double beta =		1.;	// learning rate
double epsilon =	-.001;	// MT rule parameter
double rho_a_bar =	0.;	// ARTa baseline vigilance
double  T_u =		alpha*M;      // F0 -> F2 signal to uncomited nodes.
				//	NOTE:  Changes for each input with Weber
				//	Law

#define F0_SIZE (M*2)		// Size of the F0 layer. Identical to size of the F1 layer
#define F2_SIZE MAX_F2_SIZE	// Size of the F2 layer. Identical to size of the F3 layer

/*******************************************************************************
 * Macros definition
 ******************************************************************************/
#define min(a,b)   (((a)<(b))?(a):(b))
#define max(a,b)   (((a)>(b))?(a):(b))

/*******************************************************************************
 * Function prototypes
 ******************************************************************************/
void getInputOutput(char* input, char* output, char* te_input, char* te_output);
// Loads "input," "output," "te_input", and "te_output" from their respective
//	files.  Input should be rescaled to the unit hypercube and output
//	categories should be 1-based (not 0-based because the function does
//	this for us)

void forceHypercube();
// Forces traing and testing input into unit hypercube

void checkInputOutput();
// Check if training and testing input is in the unit hypercube

void train();
// Intializes and trains the ARTMAP network on "input" and "output" (the
//	training data)

void test();
// Tests the ARTMAP network on "te_input" and "te_output" (the testing data).
//	Incorrect predictions are reported per pattern as is total number
//	correct.

void bu_match();
// Calculates T's based on the current input (A).

int is_in_Delta( int j, int Delta[], int Delta_len);
// Helper routine and returns 1 iff node j is in Delta, i.e. it is refractory

void printArr(char* str, double* arr, int len);
// Helper routine that prints "arr" with "str" prefix

void printArra(char* str, int* arr, int len);
// Helper routine that prints "arr" with "str" prefix

void printArr2(char* str, double arr[F0_SIZE][F2_SIZE], int len1, int len2);
// Helper routine that prints "arr" with "str" prefix.  Useful for tracing
//	"tau_ij"

void printArr2a(char* str, double arr[F2_SIZE][F0_SIZE], int len1, int len2);
// Helper routine that prints "arr" with "str" prefix.  Useful for tracing
//	"tau_ji"

/*******************************************************************************
 * System Setup
 *******************************************************************************/
// Singal Rule, Weber or CBD
#define DO_WEBER	0
#if DO_WEBER
    #define DO_CBD    0
#else
    #define DO_CBD	FUZZY_ARTMAP || ARTMAP_IC || ART_EMAP || DIST_ARTMAP
#endif

// Training mode.  ICG = Increased CAM Gradient
#define DO_TRAIN_WTA	FUZZY_ARTMAP || ARTMAP_IC || ART_EMAP
#define DO_TRAIN_ICG	DIST_ARTMAP

// Instance couting
#define DO_TRAIN_IC	DIST_ARTMAP || ARTMAP_IC
#define DO_TEST_IC	DIST_ARTMAP || ARTMAP_IC

// Testing mode.  SCG = Simple CAM Gradient
#define DO_TEST_WTA	FUZZY_ARTMAP
#define DO_TEST_SCG	0
#define DO_TEST_ICG	DIST_ARTMAP || ARTMAP_IC || ART_EMAP

/*******************************************************************************
 * Variables (although many of these need not be global in scope, it is easier
 *	debug if they are)
 *******************************************************************************/
double  A[F0_SIZE];     // Array representing the current input pattern (F0 layer)
double  x[F0_SIZE];     // Array representing activation at the F1 layer
double  y[F2_SIZE];     // Array representing activation at the F2 layer
double  Y[F2_SIZE];     // Array representing activation at the F3 layer
int     dist_mode;      // Variable which determines whether the system is in distributed mode
int     C;              // Number of commited F2 nodes
int     Lambda[F2_SIZE], Lambda_len, lambda;    // variables of the CAM rule index set
int     Lambda_pp[F2_SIZE], Lambda_pp_len;      // variables of the CAM rule index set in the point box case
int     Delta[F2_SIZE], Delta_len;              // Index set of F2 nodes that are refractory
int     J, K, K_prime;                          // J - index of winner in WTA mode (output class)
						// K  - index of winner in distributed mode (output class)
                                                // K' - predicted output class

double  rho;                // ARTa vigilance
int     kappa[F2_SIZE];     // Array of associations between coding nodes and output classes

double  T[F2_SIZE], S[F2_SIZE], Theta[F2_SIZE]; // Arrays used in computation of the CAM rule
double  sigma_i[F0_SIZE], sigma_k[L];               // sigma_i: signal F3 -> F1, sigma_k: signal F3 -> F0ab
double  tau_ij[F0_SIZE][F2_SIZE];               // LTM weights F0 -> F2
double  tau_ji[F2_SIZE][F0_SIZE];               // LTM weights F3 -> F1
double  c[F2_SIZE];                             // LTM weights F2 -> F3

double  Sum, aux;	    // Auxiliary variables
int     i, j, k, n, test_n, epochs;
int     cnum;		    // Number correct in testing phase

// The main function
void main()
{
  // Check System Setup
  int i=0;
  if( FUZZY_ARTMAP ) {
    i++;
  }
  if ( ARTMAP_IC ) {
    i++;
  }
  if ( ART_EMAP ) {
    i++;
  }
  if ( DIST_ARTMAP ) {
    i++;
  }

  if(i!=1)
    {
      printf("Wrong number of systems chosen. Choose exactly one system!\n");
      exit(0);
    }

  if ( (DIST_ARTMAP == 1) && (DO_CBD == 0) )
    {
      printf("dARTMAP must use CBD (check DO_CBD)\n");
      exit(0);
    }

  // Load input output.  Uncomment out this line to use the sample training/testing data
  //getInputOutput( "input.dat", "output.dat", "te_input.dat", "te_output.dat" );
  //forceHypercube(); // Uncomment this line to force unit hypercubing of input
  checkInputOutput(); // Check unit hypercubing
  // Initialize and train the network
  train();

  // Uncomment these lines if you want to see what weights the network developed
  // printArr2( "tau_ij", tau_ij, F0_SIZE, C );
  // printArr2a( "tau_ji", tau_ji, C, F0_SIZE );
  // printArr( "c", c, C );
  // printArra( "kappa", kappa, C );

  // Test the network
  test();
}

void train()
{
  printf("Training\n");

  // Initialization of the LTM weights
  for( i=0; i<F0_SIZE; i++ )
    for( j=0; j<F2_SIZE; j++ )
      {       tau_ij[i][j] = 0; tau_ji[j][i] = 0; }
  for( j=0; j<F2_SIZE; j++) c[j]=0;


 Step_1: //First iteration
  dist_mode = 1; // start in WTA mode
  n = 0;	// n is input/output index
  epochs = 0;

  // Copy the input pattern into the F1 layer with complement coding
  for( i=0; i<M; i++ )
    {
      A[i] = input[n][i];
      A[i+M] = 1-A[i];
    }
  // Copy the corresponding output class into variable K
  K =  output[n];

  C = 1; y[0]=1.; Y[0]=1.;
  for( i=0; i<F0_SIZE; i++ )  sigma_i[i]=1;
  kappa[0] = K;
  goto Step_8;

 Step_2: // Reset
  // F0 -> F2 signal
  bu_match();

  // In F2, Consider nodes whose match is above that of uncommited nodes
  Lambda_len = 0;
  for( j=0; j<C; j++ )
    if( T[j] >= T_u )
      Lambda[Lambda_len++] = j;
  if ( DO_TRAIN_WTA ) {
    dist_mode = 0;
  }

  // CAM (F2)
  // (a) If the network is in distributed mode: F2 nodes are activated
  //     according to the increased gradient CAM rule.
  if( dist_mode )
    {
      Lambda_pp_len = 0;
      for(j=0; j<Lambda_len; j++)
	if( M - T[Lambda[j]] == 0 )
	  Lambda_pp[Lambda_pp_len++] = Lambda[j];

      for( j=0; j<C; j++ )
	y[j]=0;

      // (i) If M-Tj>0 for all j belonging to Lambda...
      if( Lambda_pp_len == 0 )
	for( j=0; j<Lambda_len; j++)
	  {
	    Sum = 0;
	    for(lambda=0;lambda<Lambda_len;lambda++)
	      {
		if(Lambda[lambda]==Lambda[j])   continue;
		Sum += pow((M-T[Lambda[j]]) /
			   (M-T[Lambda[lambda]]), p);
	      }
	    y[Lambda[j]] = 1./(1.+Sum);
	  }
      // (ii) Point box case:
      else
	for( j=0; j< Lambda_pp_len; j++ )
	  y[Lambda_pp[j]] = 1./Lambda_pp_len;

      // F3 activation
      if ( DO_TRAIN_IC )
	{
	  double Sum_clyl = 0;

	  for( lambda=0; lambda<C; lambda++ )
	    Sum_clyl += c[lambda] * y[lambda];

	  for( j=0; j<C; j++ )
	    Y[j] = c[j]*y[j] / Sum_clyl;
	}
      else {
	double Sum_clyl = 0;

	for ( lambda=0; lambda<C; lambda++ )
	  Sum_clyl += y[lambda];

	for ( j=0; j<C; j++)
	  Y[j] = y[j]/Sum_clyl;
      }

      // F3 -> F1 signal
      for( i=0; i<F0_SIZE; i++ )
        {
	  double Sum=0;

	  for( j=0; j<C; j++ )
	    Sum += max(Y[j]-tau_ji[j][i],0);
	  sigma_i[i] = Sum;
        }
    }
  // (b) If the network is in WTA mode: Only one F2 node, with j=J, is
  //		activated
  else
    {
      // (i) If there is a commited node:
      if( Lambda_len > 0 )
	{
	  J = Lambda[0];
	  for( j=1; j<Lambda_len; j++ )
	    if(T[Lambda[j]]>T[J])
	      J=Lambda[j];
	}
      else    // Uncommited node
	{
	  if(C==F2_SIZE)
	    {
	      printf("No more F2 nodes available!, n = %d", n);
	      exit(0);
	    }
	  J=C;
	  C++;
	  kappa[J]=K;
	}

      // (ii) F2 and F3 activation
      for( j=0; j<C; j++ )
	y[j] = Y[j] = 0.;
      y[J] = Y[J] = 1.;

      // (iii) F3 -> F1 signal
      for( i=0; i<F0_SIZE; i++ )
	sigma_i[i] = 1-tau_ji[J][i];

      // (iv) Add J to the refractory node index set Delta
      Delta[Delta_len++]=J;
    }

 Step_3: // Reset or prediction

  // F1 activation (matching)
  Sum = 0;
  for( i=0; i<2*M; i++ )
    {
      x[i] = min( A[i], sigma_i[i] );
      Sum += x[i];
    }

  if( Sum < rho * M )
    {
      dist_mode = 0;
      //printf( "Reset! n = %d\n", n );
      goto Step_2;    // Reset
    }

 Step_4: // Prediction
  // (a) If the network is in distributed mode
  if( dist_mode )
    {
      for( k=0; k<L; k++ )
	sigma_k[k] = 0;
      for( j=0; j<C; j++ )
	sigma_k[kappa[j]] += Y[j];

      K_prime = 0;
      for( k=1; k<L; k++ )
	if( sigma_k[k]>sigma_k[K_prime] )
	  K_prime = k;
    }
  else    // If the network is in WTA mode
    K_prime = kappa[J];

 Step_5: // Match tracking or resonance
  if( K_prime == K )
    {
      if( dist_mode )
	goto Step_7; // Credit assignment
      else
	goto Step_8; // Resonance
    }

 Step_6: // Match tracking
  Sum = 0;
  for( i=0; i<2*M; i++ )
    Sum += x[i];
  rho = 1./M * Sum + epsilon;

  dist_mode = 0; //revert to WTA
  goto Step_2; // Reset

 Step_7: // Credit assignment
  // F2 blackout of incorretly predicting nodes
  for( j=0; j<C; j++ )
    if( kappa[j] != K )
      y[j]=0;

  // F2 renormalization
  Sum = 0;
  for( lambda=0; lambda<C; lambda++ )
    Sum += y[lambda];
  for( j=0; j<C; j++ )
    y[j] = y[j]/Sum;

  // F3 renormalization
  if ( DO_TRAIN_IC )
    {
      Sum = 0;
      for( lambda=0; lambda<C; lambda++ )
	Sum += c[lambda]*y[lambda];
      for( j=0; j<C; j++ )
	Y[j] = c[j]*y[j]/Sum;
    }
  else {
    Sum = 0;

    for ( lambda=0; lambda<C; lambda++ )
      Sum += y[lambda];

    for ( j=0; j<C; j++)
      Y[j] = y[j]/Sum;
  }

  // F3 -> F1 signal
  for(i=0; i<2*M; i++)
    {
      Sum = 0;
      for(j=0; j<C; j++ )
	Sum += max(Y[j]-tau_ji[j][i], 0);
      sigma_i[i] = Sum;
    }

 Step_8: //Resonance
  for( j=0; j<C; j++ )
    {
      for( i=0; i< 2*M; i++ )
	{
	  tau_ij[i][j] = tau_ij[i][j] + beta * max(y[j]-tau_ij[i][j]-A[i], 0);
	  aux = beta * max(sigma_i[i]-A[i], 0) * max( Y[j] - tau_ji[j][i], 0 );

	  if( sigma_i[i] != 0.0 ) aux = aux/sigma_i[i];

	  tau_ji[j][i] = tau_ji[j][i] + aux;
        }
      c[j] = c[j] + y[j];
    }
  Delta_len = 0; //reset node recovery
  rho = rho_a_bar; //ARTa vigilance recovery

 Step_9: // Next iteration
  if(n==TRAIN_N-1)
    {
      //Training_output
      printf("Epoch: %3d     Commited F2 nodes: %3d\n",epochs+1, C);
      if(epochs==EPOCHS-1)
	return; // End of training
      epochs++;
      n=-1;
    }
  n++;

  // Copy next input pattern and corresponding output category
  for( i=0; i<M; i++ )
    {
      A[i] = input[n][i];
      A[i+M] = 1-A[i];
    }
  K = output[n];
  dist_mode = 1; //revert to distributed mode
  goto Step_2;
}

void test()
{
  printf("Testing\n");
  cnum = 0;

 Test_Step_1: //First iteration
  for(test_n = 0; test_n < TEST_N; test_n++)
    {
      // Copy next input pattern and corresponding output category
      for( i=0; i<M; i++ )
        {
	  A[i] = te_input[test_n][i];
	  A[i+M] = 1-A[i];
	}
      K = te_output[test_n];

    Test_Step_2:  // Reset
      bu_match();

      if ( DO_TEST_ICG ) {
	Lambda_len = 0;
	for( j=0; j<C; j++ )
	  if( T[j] >= T_u )
	    Lambda[Lambda_len++] = j;

	// (a) If the network is in distributed mode: F2 nodes are activated
	//     according to the increased gradient CAM rule.
	Lambda_pp_len = 0;
	for(j=0; j<Lambda_len; j++)
	  if( M - T[Lambda[j]] == 0 )
	    Lambda_pp[Lambda_pp_len++] = Lambda[j];

	for( j=0; j<C; j++ )
	  y[j]=0;

	// (i) If M-Tj>0 for all j belonging to Lambda...
	if( Lambda_pp_len == 0 )
	  for( j=0; j<Lambda_len; j++)
	    {
	      Sum = 0;
	      for(lambda=0;lambda<Lambda_len;lambda++)
                {
		  if(Lambda[lambda]==Lambda[j])   continue;
		  Sum += pow((M-T[Lambda[j]]) /
			     (M-T[Lambda[lambda]]),p);
                }
	      y[Lambda[j]] = 1./(1.+Sum);
	    }
	// (ii) Point box case:
	else
	  for( j=0; j< Lambda_pp_len; j++ )
	    y[Lambda_pp[j]] = 1./Lambda_pp_len;

	// F3 activation
	if ( DO_TEST_IC )
	  {
            double Sum_clyl = 0;
            for( lambda=0; lambda<C; lambda++ )

	      Sum_clyl += c[lambda] * y[lambda];
            for( j=0; j<C; j++ )
	      Y[j] = c[j]*y[j] / Sum_clyl;
	  }
	else {
	  double Sum_clyl = 0;

	  for ( lambda=0; lambda<C; lambda++ )
	    Sum_clyl += y[lambda];

	  for ( j=0; j<C; j++)
	    Y[j] = y[j]/Sum_clyl;
	}
      }

      if ( DO_TEST_SCG ) {
	Sum = 0;
	for ( i = 0; i < C; i++)
	  Sum += pow(T[i], p);

	for( i = 0; i < C; i++)
	    T[i] = pow(T[i],p)/Sum;

	// F2 activation
	Sum = 0;
	for ( i = 0; i < C; i++)
	  Sum += T[i];

	for ( i = 0; i < C; i++)
	  y[i] = T[i]/Sum;

	// F3 activation
	if ( DO_TEST_IC ) {
	  double Sum_clyl = 0;
	  for( lambda=0; lambda<C; lambda++ )
	    Sum_clyl += c[lambda] * y[lambda];

	  for( j=0; j<C; j++ )
	    Y[j] = c[j]*y[j] / Sum_clyl;
	}
	else {
	  double Sum_clyl = 0;

	  for ( lambda=0; lambda<C; lambda++ )
	    Sum_clyl += y[lambda];

	  for ( j=0; j<C; j++)
	    Y[j] = y[j]/Sum_clyl;
	}
      }

      if ( DO_TEST_WTA ) {
	// (b) If the network is in WTA mode: Only one F2 node, with j=J, is activated
	// (i) If there is a commited node:
	if( Lambda_len > 0 )
	  {
	    J = Lambda[0];
	    for( j=1; j<Lambda_len; j++ )
	      if( T[Lambda[j]] > T[J])
		J=Lambda[j];
	  }
	else    // No "commited" nodes, but pick the best you can
	  {
	    double Tmax = -1.0;
	    for (j = 1; j < C; j++) {
	      if ( T[j] > Tmax ) {
		Tmax = T[j];
		J = j;
	      }
	    }
	  }

	// (ii) F2 and F3 activation
	for( j=0; j<C; j++ )
	  y[j] = Y[j] = 0.;
	y[J] = Y[J] = 1.;
      }

Test_Step_3: // Prediction
        for( k=0; k<L; k++ )
	  sigma_k[k] = 0;
        for( j=0; j<C; j++ )
	  sigma_k[kappa[j]] += Y[j];

        K_prime = 0;
        for( k=1; k<L; k++ )
	  if( sigma_k[k] > sigma_k[K_prime] )
	    K_prime = k;

Test_Step_4: // Evaluation
        if( K_prime == K )
	  {
            cnum++;
	  }
        else
	  //Testing_output
	  printf("Test pat #%4d: INcorrect prediction\n", test_n+1);
    }
  printf("Number of correctly classified test patterns: %d\n",cnum);
}

int is_in_Delta( int j, int *Delta, int Delta_len)
{
  int in_Delta = 0;

  for(int  in_Delta_index = 0; in_Delta_index < Delta_len; in_Delta_index++ )
    if( j == Delta[in_Delta_index] )
      {
            in_Delta = 1;
            break;
      }
  return in_Delta;
}

void getInputOutput(char* train_input, char* train_output, char* test_input, char* test_output) {
    // read training and testing data from files
    FILE* input_f     = fopen(train_input,"r");
    FILE* output_f    = fopen(train_output,"r");
    FILE* te_input_f  = fopen(test_input,"r");
    FILE* te_output_f = fopen(test_output,"r");
    int i, j;

    if( ! (input_f && output_f && te_input_f && te_output_f) )
    {
      printf("Error: Unable to open one of the data files!\n");
      exit(0);
    }

    for( i=0; i<TRAIN_N; i++ )
      for(int j=0; j<M; j++ )
	fscanf(input_f,"%lf",&(input[i][j]));

    for( i=0; i<TRAIN_N; i++ )
      {
        fscanf(output_f,"%d",&output[i]);
        output[i] = output[i] - 1;
      }

    for( i=0; i<TEST_N; i++ )
      for(int j=0; j<M; j++ )
	fscanf(te_input_f,"%lf",&(te_input[i][j]));

    for( i=0; i<TEST_N; i++ )
      {
        fscanf(te_output_f,"%d",&(te_output[i]));
        te_output[i] = te_output[i] - 1;
      }

    fclose(input_f);
    fclose(output_f);
    fclose(te_input_f);
    fclose(te_output_f);
}

void forceHypercube() {
  double max[M], min[M];
  for ( int i = 0; i < M; i++ ) {
    max[i] = input[0][i];
    min[i] = input[0][i];
  }

  for ( int i = 0; i < TRAIN_N; i++ ) {
    for ( int j = 0; j < M; j++) {
      if (input[i][j] > max[j] )
	max[j] = input[i][j];

      if (input[i][j] < min[j] )
	min[j] = input[i][j];
    }
  }

  for ( int i = 0; i < TRAIN_N; i++ ) {
    for ( int j = 0; j < M; j++) {
      input[i][j] = (input[i][j] -  min[j]) / (max[j]-min[j]);
    }
  }

  for ( int i = 0; i < M; i++ ) {
    max[i] = te_input[0][i];
    min[i] = te_input[0][i];
  }

  for ( int i = 0; i < TEST_N; i++ ) {
    for ( int j = 0; j < M; j++) {
      if (te_input[i][j] > max[j] )
	max[j] = te_input[i][j];

      if (te_input[i][j] < min[j] )
	min[j] = te_input[i][j];
    }
  }

  for ( int i = 0; i < TEST_N; i++ ) {
    for ( int j = 0; j < M; j++) {
      te_input[i][j] = (te_input[i][j] -  min[j]) / (max[j]-min[j]);
    }
  }

}

void checkInputOutput() {
  int quit = 0;

  for ( int i = 0; i < TRAIN_N && !quit; i++ ) {
    for ( int j = 0; j < M && !quit; j++) {
      if (input[i][j] > 1.0 || input[i][j] < 0.0)
	quit = 1;
    }
  }

  if ( quit == 1 ) {
    printf( "Training input is not in unit hypercube!\n" );
    exit(0);
  }

  quit = 0;

  for ( int i = 0; i < TEST_N && !quit; i++ ) {
    for ( int j = 0; j < M && !quit; j++) {
      if (te_input[i][j] > 1.0 || te_input[i][j] < 0.0)
	quit = 1;
    }
  }

  if ( quit == 1 ) {
    printf( "Testing input is not in unit hypercube!\n" );
    exit(0);
  }
}

void bu_match() {
  for( j=0; j<C; j++ )
    {
      if (is_in_Delta( j, Delta, Delta_len) )
	T[j] = 0;
      else
        {
	  if ( DO_CBD ) {
            // Choice by difference function
            S[j] = 0; Theta[j] = 0;
            for( i=0; i<2*M; i++ )
	      {
                S[j] += min(A[i], 1-tau_ij[i][j]);
                Theta[j] += tau_ij[i][j];
	      }
            T[j] = S[j]+(1-alpha)*(Theta[j] - M);
	  }

	  if ( DO_WEBER ) {
            // Weber's law choice function
            T[j]=Sum=0;
            for( i=0; i<2*M; i++ )
	      {
                T[j] += min(A[i], 1-tau_ij[i][j]);
                Sum += 1-tau_ij[i][j];
	      }
            T[j] = T[j] / (alpha + Sum );

	    T_u = M/(alpha + 2.0*M);
	  }

        }
    }
}

void printArr( char* str, double* arr, int len) {
  int i;

  fprintf( stdout, "**\t%s = ", str );
  for (i = 0; i < len-1; i++) {
    fprintf( stdout, "%5.2f, ", arr[i] );
  }
  fprintf( stdout, "%5.2f\n", arr[i] );
}

void printArra( char* str, int* arr, int len) {
  int i;

  fprintf( stdout, "**\t%s = ", str );
  for (i = 0; i < len-1; i++) {
    fprintf( stdout, "%d, ", arr[i] );
  }
  fprintf( stdout, "%d\n", arr[i] );
}

void printArr2( char* str, double arr[F0_SIZE][F2_SIZE], int len1, int len2) {
  int i, j;

  for (i = 0; i < len1; i++) {
    fprintf( stdout, "**\t%s[%d] = ", str, (i+1) );
    for (j = 0; j < len2-1; j++) {
      fprintf( stdout, "%5.2f, ", arr[i][j] );
    }
    fprintf( stdout, "%5.2f\n", arr[i][j] );
  }
}

void printArr2a( char* str, double arr[F2_SIZE][F0_SIZE], int len1, int len2) {
  int i, j;

  for (i = 0; i < len1; i++) {
    fprintf( stdout, "**\t%s[%d] = ", str, (i+1) );
    for (j = 0; j < len2-1; j++) {
      fprintf( stdout, "%5.2f, ", arr[i][j] );
    }
    fprintf( stdout, "%5.2f\n", arr[i][j] );
  }
}
