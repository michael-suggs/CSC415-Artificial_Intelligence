/********************* FUNCTS.C ************************************/

#include "std_fun.h"

/*-------------------------- rect -----------------------*/

double rect(x)  /* This functions rectifies x at 0.0 */
     double x;
{
  return((x > 0.0) ? x : 0.0);
}


/*---------------------------  HVS -------------------------*/

double HVS(x)  /* This is the standard Heavyside function */
     double x;
{
  return((x > 0.0) ? 1.0 : 0.0);
}


/*---------------------------- f ---------------------------*/
/* The infamous feedback signal function: here threshold-linear */

double f(x,sigma)
     double x, sigma;
{
    return((x > sigma) ? x : 0.0);
}



/*---------------------------- f ---------------------------*/
/* A different version of f above: threshold linear, with zero
 activation for x = a */

double fab(x,a,b)
     double x,a,b;
{
    x=(x > b) ? b : x;			/* chop off the top */
    return((x > a) ? x-a : 0.0);	/* chop off the bottom */
}



/*--------------------------- L2_norm -------------------------*/
/* This function returns the L2 norm of a vector of any dimension */

double L2_norm(vec,dim)
     double *vec;
     short dim;
{
    int i;
    double t_sum=0.0;

    for (i=0; i<dim; i++)
      t_sum += vec[i]*vec[i];

    return(sqrt(t_sum));
}


/*----------------------------- vec_diff ------------------------------*/
/* This function returns the difference between two vectors of arbitrary
   dimension. The absolute value of the difference between each element
   is added up, and the total is divided by the dimension. */

double vec_diff(v1,v2,dim)
     double *v1, *v2;
     int dim;
{
    int i;
    double d_mag=0.0;

    for (i=0; i<dim; i++)
      d_mag += fabs(v1[i]-v2[i]);

    return(d_mag/(double)dim);
}
