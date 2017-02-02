/************************************ in_vec.c ***************************/

#include <stdio.h>
#include "in_vec.h"
#include "art2.h"

FILE *datain, *fopen();
int p_dim, p_num, *IL;
double **I, *p_mag;
int ppc, *c_p;
extern char fname[];



/***************************** get_inputs *******************************/
/* Here we open the input file, and read in the input patterns. The function
   allocates as much space as needed for the patterns. */

void get_inputs()
{
    int i,j;
    char in_name[80];

    if (fname[0] == '\0') {
	fprintf(stderr,"Please enter input file name (no extension) --> ");
	scanf("%s",fname);
    }
    sprintf(in_name,"%s.inp",fname);
    if ((datain=fopen(in_name,"r")) == NULL) {
	fprintf(stderr,"Fatal Error: Cannot open %s\n",in_name);
	exit(1);
    }

    fprintf(stderr,"\n********************** reading %s\n\n",in_name);

    fscanf(datain,"%d %d",&p_dim,&p_num);
    fprintf(stderr,"Allocating and reading %d %d-D vectors\n",p_num,p_dim);

    /* allocate enough space */
    if ((I=(double **) calloc(p_num,sizeof(double *))) == NULL) {
	fprintf(stderr,"Error: could not allocate %d %d-D patterns!\n",
		p_num,p_dim);
	exit(0);
    }

    for (i=0; i<p_num; i++)
      if ((I[i]=(double *) calloc(p_dim,sizeof(double))) == NULL) {
	  fprintf(stderr,"Error: could not allocate %d %d-D patterns!\n",
		  p_num,p_dim);
	  exit(0);
      }

    if ((p_mag=(double *) calloc(p_num,sizeof(double))) == NULL) {
	fprintf(stderr,"Error: could not allocate %d %d-D patterns!\n",
		p_num,p_dim);
	exit(0);
    }

    if ((IL=(int *) calloc(p_num,sizeof(int))) == NULL) {
	fprintf(stderr,"Error: could not allocate %d %d-D patterns!\n",
		p_num,p_dim);
	exit(0);
    }



    /* do bookkeeping for new learning set */
    new_learning();

    /* get patterns */
    for (i=0; i<p_num; i++)
      {
	  p_mag[i] = 0.0;
	  for (j=0; j<p_dim; j++)
	    if (fscanf(datain,"%lf",I[i]+j) == EOF) {
		fprintf(stderr,"Error: Unexpected End-Of-File in %s\n",
			in_name);
		exit(1);
	    }
	  p_mag[i] = L2_norm(I[i],p_dim); /* Keep track of pattern magnitude */
	  if (p_mag[i] <= 0.0)  /* check for all zeros! */
	    {
		fprintf(stderr,"(Warning: pattern %d has mag=0.0)",i);
		p_mag[i] = 1.0;
	    }
	  printf(". ");
      }

    /* now get dimensions for network. If no dimensions exist assign
       M=p_dim, N=M+p_num by default. */
    if (fscanf(datain,"%d %d",&M,&N) != EOF) {
	if (M < p_dim)
	  fprintf(stderr,"Warning: Too few F0-1 nodes: changed M to %d\n",
		  (M=p_dim));
	if (N <= M)
	  fprintf(stderr,"Warning, requested only %d total nodes. Assigning %d\n",
		  N,N=M+p_num);
    }
    else {
	M=p_dim;
	N=M+p_num;
	fprintf(stderr,"No values for M and N found. Assigning M=%d and N=%d\n"
		,M,N);
    }


    fclose(datain);
    printf("done.\n");
}





/***************************** select_one *******************************/
/* This function is called if the user wants to select a single pattern
   for presentation to ART. */

void select_one(i)
     int i;
{
    do
      {
	  fprintf(stderr, "select pattern for cycle %d --> ",i);
	  scanf("%d",&c_p[i]);
	  if (c_p[i] >= p_num || c_p[i] < 0)
	    fprintf(stderr,"Invalid choice. Try again.\n");
      } while (c_p[i] >= p_num || c_p[i] < 0);
}

/***************************** select_cycle *******************************/
/* Allow the user to select a number of patterns to be presented to ART
   during a learning cycle. */

void select_cycle(num)
     short num;
{
    int i;

    if (num < 1)
      {
	  fprintf(stderr, "How many patterns per cycle? ");
	  scanf("%d",&ppc);
      }
    else
      ppc = num;

    if (c_p != NULL)
      free(c_p);

    if ((c_p=(int *) calloc(ppc,sizeof(int))) == NULL) {
	fprintf(stderr,"Error: could not allocate %d integers.\n",ppc);
	exit(0);
    }

    /* select each pattern individually. This could be automated for
       specific applications by reading another input file or some such. */
    for (i=0; i<ppc; i++)
      select_one(i);
}


/***************************** init_all *******************************/
/* Initialize everything */

void init_all()
{
    int i,j;
    void free_all(), get_inputs();

    free_all(); /* free everything that is allocated. */
    BU_reset(); /* initialize weights */
    TD_reset();
    get_inputs(); /* get new input file */

}


/***************************** new_learning *******************************/
/* Minor bookeeping for new learning cycle inputs. */

void new_learning()
{
    int i;

    for (i=0; i<p_num; i++)
      IL[i] = -1;
}


/***************************** free_all *******************************/
/* de-allocate all allocated pointers */

void free_all()
{
    int i;

    for (i=0; i<p_num; i++)
      free(I[i]);
    free(I);
    free(IL);
    free(p_mag);

    if (c_p != NULL)
      free(c_p);
}


/***************************** p_display *******************************/
/* Display the selected pattern on the screen */

void p_display(p_n)
     int p_n;
{
  int i,j;

  fprintf(stderr,"pattern %d (%d-D): ",p_n,p_dim);
  for (i=0; i<p_dim; i++)
    {
	fprintf(stderr,"%4.3lf  ",I[p_n][i]);
    }
  fprintf(stderr,"\n");

}
