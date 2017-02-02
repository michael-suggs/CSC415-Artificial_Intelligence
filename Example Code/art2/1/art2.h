/************************************* art2.h ******************************/

#include <stdio.h>
#include <math.h>
#include "rk4d.h"
#include "in_vec.h"

extern int M, N; /* Note: for historical reasons this entire program is
		    based on the assumption that there are M nodes at F1
		    and (N-M) nodes at F2. Hence N represents the total
		    number of nodes in F1 and F2. */

extern double rho, a, b, c, d, theta, ac_dif;

extern FILE *fout, *flog;

extern double h;	/* Runge-Kutta step size */
extern int nsteps; /* Number of time steps */

extern double mag_X;
extern int F2_on;

