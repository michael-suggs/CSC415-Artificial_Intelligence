/******************************** art2 **************************************

This program is a simulation of the ART2 architecture. Done as simulation
assignment numbers 3&4 for CN550.  Oct. 23, 1989.
Revised for Assignment 4 on Nov. 4, 1989.
Cleaned up on Mar. 13, 1990.

NOTE: The architecture implemented here is based on the following article:

G. Carpenter and S. Grossberg (1987) ``ART 2: self-organization of
stable category recognition codes for analog input patterns.'' {\em
Applied OpticsL,\/ {\bf 26L, (23), 4919-4930.

The particular version implemented here is like that shown in figure
10. In addition, the F0 loop is made isomorphic to the F1 layer, i.e.,
it consists of two loops and not just one as shown in the figure. This
has the advantage of buffering the input from TD influence. The
equations change a little bit in some cases, but I tried to document
these changes in the code as needed. Note that the bottom loop is
referred to as F0, the two middle loops as F1, and the top loop as F2.
The learning rule used here is the slow learning rule, meaning that
weights are updated by integrating the appropriate differential
equation (eqs. 18 & 19 in the article). The fast learning rule is good
for theoretical analyses, but not very good for realistic
categorization, particularly with ART2.

There are some tricks to implementing the circuit: 1) Regardless of
what kind of learning you use (fast or slow), the node activations are
always assumed to be fast, i.e., you solve them at steady state.
However, this requires that the activation in the two loops of F1 be
iterated constantly as the LTM values are updated. This is because
changes in the Bottom-Up (BU) or Top-Down (TD) LTM traces will change
the steady state response of the F1 populations.  2) it is suggested
that every time a reset takes place at F2, all F1 and F0 nodes should
also be reset. This avoids reverberations from old patterns that can
badly deform new inputs. The ART3 architecture avoids this problem
altogether.

Note that in my simulation, I let the weights change at the end of
every iteration through F1. The alternative approach would be to
iterate through F1 until activation has settled, then change weights,
iterate F1 until it settles, then update weights, and so on. This is
simply my interpretation of the fact that the weights, albeit slower
than STM activations, are still only finitely slower than STM. The
effects are minor. Mostly what you see is that after learning has
stabilized for a given pattern, a subsequent presentation of the same
pattern will cause a slight transient shift in the weights. This is
because as the pattern is instated at the top node P of F1, it takes a
few cycles before the feedback loop amplifies it to its steady-state
value. However, because F2 and the LTM traces are continuously
updated, the Top-Down exepctation pattern is parallel to the vector P
(hence no reset takes place), but it expects a bigger amplitude. Hence
the weights drop slightly, and then grow back to their peak values as
the F1 loops settle down. The size of the "dip" will be a measure of
the relative rate of LTM and STM changes. If you make the LTM traces
update more slowly (by reducing the step size of integration), the dip
will be smaller, and vice versa.

Mar 27 note: I have gone ahead and changed the function update_Zs() to
use an equation that should avoid the "dip" problem. I found out that
under extreme conditions the other version could cause a reset during
learning, which is a BAD thing.

ABOUT THE OUTPUT: The output of this program is very crude. Basically I
left it wide open so users could make any changes they wanted to suit whatever
output viewing devices they have. I have placed comments were appropriate
to indicate good places for printing out data.


*****************************************************************************/

#include "art2.h"

int N, M;

double *X, *Xp, *W, *Wp, *U, *Up, *P, *Pp, *V, *Vp, *Q, *Qp, *R, *Y, *mis;
double **Zij, **Zji;
double *old1M, *old2M; /* these are used as temp variables */

int n_mis=0;	/* number of mismatched nodes */

/* recommended initial parameters. */
double rho=0.9, a=5.0, b=5.0, c=0.225, d=0.8,
  theta=0.3,ac_dif=0.001,z_dif=0.01;

/* two output files: one for a running log, the other one for data */
FILE *fout, *flog;
char fname[80], out_name[80], log_name[80];

double h=0.1;	/* Runge-Kutta step size */
int nsteps=100; /* Number of time steps */

/* Miscellaneous flags */

double mag_X;

int F2_on; /* which category is active? Note that because C starts counting at
	      zero, F2_on = -1 is the no-choice value. */

int l_flag; /* learning or not. No learning is only used for testing, e.g.,
	       to see where a new node *would be* categorized. */

int rsnc=0; /* Resonance flag */


/********************************* MAIN *********************************/

main(argc,argv)
     int argc;
     char **argv;
{
  int i, choice;
  void do_cycle(), alloc_pops(), free_pops();
  void par_modify(), show_setup(), init_weights();
  void BU_reset(), TD_reset();
  char tname[20];
  char tmp[10];

  if (argc > 1)
    strcpy(fname,*++argv);
  else
    strcpy(fname,"");

  get_inputs(); /* this does initialization. See in_vec.[ch] */

  alloc_pops();

  if (argc < 3) {
      fprintf(stderr,"Enter output filename (no extension) -> ");
      scanf("%s",tname);
  }
  else
      strcpy(tname,*++argv);

  sprintf(out_name,"%s.dat",tname);
  sprintf(log_name,"%s.log",tname);
  fout = fopen(out_name,"w");
  flog = fopen(log_name,"w");

  /* Initialize all weigths. ART will not work if BU weights are zero. */
  BU_reset();
  TD_reset();

  fprintf(flog,"*****> %s\n",log_name); /* Mark the file */
  fprintf(flog,"Input file is %s.inp\n",fname);

  do /* MAIN MENU */
    {
	fprintf(stderr,"\n***\nSelect one option:\n");
	fprintf(stderr,"1) Select input pattern\n");
	fprintf(stderr,"2) Select a series of patterns\n");
	fprintf(stderr,"3) Cycle input pattern (no learning)\n");
	fprintf(stderr,"4) Cycle input pattern (slow learning)\n");
	fprintf(stderr,"5) Modify a parameter\n");
	fprintf(stderr,"6) Initialize weights\n");
	fprintf(stderr,"7) Reset Learning flag\n");
	fprintf(stderr,"8) Show setup\n");
	fprintf(stderr,"9) Reset everything\n");
	fprintf(stderr,"Other) Quit\n\n");
	fprintf(stderr,"Enter number of your choice --> ");
	scanf("%d",&choice);

	switch (choice) {
	  case 1 : select_cycle(1);
	    break;
	  case 2 : select_cycle(0);
	    break;
	  case 3 : l_flag=0;
	    do_cycle();
	    break;
	  case 4 :
	    l_flag=1;
	    do_cycle();
	    break;
	  case 5 : par_modify();
	    break;
	  case 6 : init_weights();
	    break;
	  case 7 : new_learning();
	    break;
	  case 8 : show_setup();
	    break;
	  case 9 : init_all();
	    break;
	  default : /* make sure it is not an error */
	    fprintf(stderr,"Do you really want to exit (y/n)? ");
	    scanf("%s",tmp);
	    if (tmp[0] == 'Y' || tmp[0] == 'y')
	      choice = 0;
	    else
	      choice = -1;
	    break;
	}

    } while (choice != 0);

  /* wrap it up. */
  free_all();
  fclose(fout);
  fclose(flog);

}


/***************************** do_cycle *******************************/
/* This is the heart of the program. It is the main loop that runs through
   all the iterations. It is used both in learning and no-learning modes. */

void do_cycle()
{
    int i,j;
    void update_F0(), update_F1(), update_F2(), print_weights();
    double update_R();

    if (l_flag)
      fprintf(flog,"\nStarting a learning cycle.\n");
    else
      fprintf(flog,"\nStarting a cycle with no learning.\n");

    for (j=0; j<N-M; j++) /* Initially, no mismatches yet! */
      mis[j]=0;
    n_mis=0;
    /* note: doing the above reset OUTSIDE of the subsequent loop reflects
       the hypothesis that the reset of an F2 node lasts as long as the
       entire learning trial. This means that an F2 node that has been reset
       will not be able to encode a subsequent correct pattern IN THE SAME
       TRIAL. This point is entirely arbitrary, and it could be changed so
       that resets can only affect a single pattern. Either solution leads
       to peculiar dynamics, and the only "correct" solution is the one
       adpoted by ART3, where the reset is actually a dynamic process. */

    for (i=0; i<ppc; i++)
      {
	  for (j=0; j<M; j++) {	     /* reset F0, F1 nodes for new pattern */
	      Wp[j]=Xp[j]=Up[j]=Vp[j]=Pp[j]=Qp[j]=0.0;
	      W[j]=P[j]=X[j]=Q[j]=V[j]=U[j]=0.0;
	  }
	  F2_on=-1; /* No F2 node active from before */

	  fprintf(stderr,"***\nNo.%d (of %d) in cycle is input pattern %d\n",
							i,ppc,c_p[i]);
	  fprintf(flog,"***\nNo.%d (of %d) in cycle is input pattern %d\n",
							i,ppc,c_p[i]);
	  update_F0(c_p[i]); /* self-explanatory */
	  update_F1(c_p[i]);
      }

    for (i=0; i<ppc; i++) { /* show final configuration */
	fprintf(stderr,"Pattern %d is coded at F2 node %d\n",c_p[i],
		IL[c_p[i]]);
	fprintf(flog,"Pattern %d is coded at F2 node %d\n",c_p[i],
		IL[c_p[i]]);
    }
    fprintf(stderr,"\n");
    fprintf(flog,"\n");

    /* if learning, print out the weights. I have taken it out here,
       and do it manually from the main menu. If you want automatic weight
       printout at the end of each complete cycle, uncomment the next lines
       */

/*    if (l_flag)
      print_weights(); */
}


/******************* updating functions **********************************/


/***************************** update_F0 *******************************/
/* This function updates the F0 nodes until a stable activity level is
   reached. Stability is checked only at population Wp, since this is the
   one that is most likely to change. */

void update_F0(p)
     int p; /* The input pattern */
{
    int i,cntr=0;
    double *old_W = old1M;
    double ptmp,wtmp,vtmp;
    double v_d=1.0; /* Magnitude of the vector change in Wp. See below */

    do /* keep going until stable */
      {
	  for (i=0; i<M; i++) /* Pass I: update Wp, and Pp */
	    {
		old_W[i]=Wp[i];
		Wp[i]=I[p][i]+a*Up[i];
		Pp[i]=Up[i];
	    }
	  ptmp=L2_norm(Pp,M); /* calculate norms */
	  wtmp=L2_norm(Wp,M);

	  /* IMPORTANT: If the norm of Pp or Wp is zero (or thereabouts) the
	     normalizing constant should be 1 to avoid blowups in Qp, Xp,
	     and Vp below. */
	  ptmp = (ptmp > 0.1) ? ptmp : 1.0;
	  wtmp = (wtmp > 0.1) ? wtmp : 1.0;

	  for (i=0; i<M; i++) /* Pass II: update Qp, Xp, and Vp. */
	    {
		Qp[i]=Pp[i]/ptmp;
		Xp[i]=Wp[i]/wtmp;
		Vp[i]=f(Xp[i],theta)+b*f(Qp[i],theta);
	    }

	  vtmp=L2_norm(Vp,M);
	  vtmp = (vtmp > 0.1) ? vtmp : 1.0; /* Again, avoid norm of zero */

	  for (i=0; i<M; i++) /* Pass III, calculate Up */
	    {
		Up[i]=Vp[i]/vtmp;
	    }

	  cntr++; /* keep track of how many iterations! */

	  v_d = (vec_diff(old_W,Wp,M)); /* change in Wp from last cycle */

      } while ((v_d > ac_dif && cntr < nsteps) || cntr < 2);
    /* check if: (1) Wp activation has converged within ac_dif, or (2) we have
       taken more than nsteps iterations. The cntr < 2 is needed for funny
       cases when convergence was spuriously achieve on the first cycle.
       This can't be avoided in all the possible cases. */

    /* Dump out all F0 information */
    fprintf(flog,"F0 activity (W') has stabilized within %5.4lf after %d cycles.\n\n",
	    v_d,cntr);

    fprintf(flog,"  Wp    Pp     Xp     Qp     Vp     Up\n");

    for (i=0; i<M; i++)
      fprintf(flog,"%4.3lf  %4.3lf  %4.3lf  %4.3lf  %4.3lf  %4.3lf\n",
	      Wp[i],Pp[i],Xp[i],Qp[i],Vp[i],Up[i]);
    fprintf(flog,"\n");
}





/***************************** update_F1 *******************************/
/* The biggie. Actually updates F2 while it updates F1. See main comment
   header at top of program. */

void update_F1(p)
     int p;
{
    int i,cntr=0;
    int pf; /* print flag: how often to print (see below) */
    double ptmp,wtmp,vtmp;
    double *old_W=old1M, *old_P=old2M;/*here we keep track of two
 populations*/
    double v_d = 1.0; /* vector differences */
    double z_d = 0.0;

    double update_R(), update_Zs();
    void update_F2(), reset_F2();

    do
      {
	  pf = (cntr%10==0); /* print every 10 steps. This number should be
				changed by looking in a config file, so no
				recompile is needed */

	  for (i=0; i<M; i++)  /* update W and P */
	    {
		old_W[i]=W[i];
		old_P[i]=P[i];
		W[i]=Qp[i]+a*U[i];
		ptmp=F2_on < 0 ? 0.0 : d*Zji[F2_on][i];
		P[i]=U[i]+ptmp;
	    }
	  ptmp=L2_norm(P,M);
	  wtmp=L2_norm(W,M);

	  /* IMPORTANT: If the norm of P or W is zero (or thereabouts) the
	     normalizing constant should be to avoid blowups in Q, X,
	     and V below. */
	  ptmp = (ptmp > 0.1) ? ptmp : 1.0;
	  wtmp = (wtmp > 0.1) ? wtmp : 1.0;

	  for (i=0; i<M; i++)  /* update Q, X, V */
	    {
		Q[i]=P[i]/ptmp;
		X[i]=W[i]/wtmp;
		V[i]=f(X[i],theta)+b*f(Q[i],theta);
	    }

	  vtmp=L2_norm(V,M);
	  vtmp = (vtmp > 0.1) ? vtmp : 1.0; /* again, avoid norm=0 */

	  for (i=0; i<M; i++) /* update U */
	    {
		U[i]=V[i]/vtmp;
	    }

	  /* Here we check changes in P as well as W. This is because both
	     of these populations receive inputs from outside F1: W gets
	     input from F0, and P gets inputs from F2. */
	  v_d=vec_diff(old_W,W,M)+vec_diff(old_P,P,M);

/* Now update weights (if necessary) */
	  z_d=0.0;

	  if (!rsnc) /* if NOT resonating, update F2. Note that this should
			not be necessary, i.e., updating F2 should be harmless.
			However, it saves computational time here. */
	    update_F2(cntr);
	  else {
	      /* if it just started resonating, print a blurb */
	      if (cntr < 3) {
		  fprintf(stderr,"%d) %d is resonating on F2 node %d\n",cntr,
			  p,F2_on);
		  fprintf(flog,"%d) %d is resonating on F2 node %d\n",cntr,
			  p,F2_on);
	      }
	      IL[p]=F2_on; /* This keeps track of which node codes pattern p */
	  }

	  if (update_R(cntr) < rho) { /* check for mismatch at R! */
	      reset_F2(p,cntr);
	      cntr=0;
	      rsnc=0;
	  }
	  else {
	      /* no reset: go ahead and learn, unless in no-learning mode.
		 Also, there may be NO active F2 nodes, in which case don't
		 learn. */
	      if (l_flag > 0 && F2_on > -1)
		z_d=update_Zs(cntr);
	  }


	  if (n_mis == N-M) { /* mismatched at all F2 nodes! */
	      fprintf(stderr,"%d) All F2 nodes have been reset by input %d!\n",
		      cntr,p);
	      fprintf(flog,"%d) All F2 nodes have been reset by input %d!\n",
		      cntr,p);
	      break; /* get out of loop */
	  }

	  cntr++; /* Keep track of iterations. */

      } while ((v_d > ac_dif && cntr < nsteps) || cntr < 2 || z_d > z_dif);
    /* Check that: activation has stabilized, we have not taken too many
       steps, and weights have converged. cntr < 2 takes care of weird cases
       that look like convergence on first step. */

    rsnc=0; /* reset resonance flag. */

    /* Print out F1 and R STM values. */
    fprintf(flog,"F1 activity has stabilized within %lf after %d cycles.\n",
	    ac_dif,cntr);
    fprintf(stderr,"F1 activity has stabilized within %lf after %d cycles.\n",
	    ac_dif,cntr);

    fprintf(stderr,"   W      P      X      Q      V      U      R\n");
    fprintf(flog,"   W      P      X      Q      V      U      R\n");

    for (i=0; i<M; i++) {
	fprintf(stderr,"%4.3lf  %4.3lf  %4.3lf  %4.3lf  %4.3lf  %4.3lf %4.3lf\n"
		,W[i],P[i],X[i],Q[i],V[i],U[i],R[i]);
	fprintf(flog,"%4.3lf  %4.3lf  %4.3lf  %4.3lf  %4.3lf  %4.3lf  %4.3lf\n"
		,W[i],P[i],X[i],Q[i],V[i],U[i]);
    }
    fprintf(stderr,"*** F1 diff = %lf\n\n",v_d);
    fprintf(flog,"*** F1 diff = %lf\n\n",v_d);
}





/***************************** update_F2 *******************************/
/* As it says in the name, this funciton updates F2 values */

void update_F2(cntr)
     int cntr;
{
    int i,j;
    double ymax; /* which node has maximum activation? */
    int pf = (cntr%10==0); /* how often to print. */

    ymax=0.0;
    for (j=0; j<N-M; j++) { /* find winner at F2. */
	Y[j]=0.0;
	if (mis[j] > -1)  /* do not update reset nodes */
	  for (i=0; i<M; i++)
	    Y[j] += Zij[i][j]*P[i];
	ymax = (ymax < Y[j] ? Y[j] : ymax);
      }

/* NOTE: The above method of chosing a winner has two problems: 1) it is
   order-dependent, meaning that if many F2 nodes have exactly the same
   activation, the one with lowest index wins. This seems arbitrary, but
   if the initial weights are even slightly random then it is very unlikely
   that it would matter anyway. 2) It is extremely important to make sure that
   at least one F2 node is active. Sometimes it is possible to setup F0
   thresholds (theta) so high that no patterns come through. If this function
   blindly selects the first winner, it will select node zero even though
   no input is present. Then the F2 node will have an activation of 1.0,
   and you will magically get activation in F1 even though it does not belong
   there. I avoid this by forcing F2_on = -1 if all nodes are zero. */

    if (ymax == 0.0) {  /* No coding anywhere */
	fprintf(stderr,"%d) All F2 nodes are inactive! No category chosen.\n",
		cntr);
	fprintf(flog,"\n%d) All F2 nodes are inactive! No category chosen.\n",
		cntr);
	F2_on = -1;
	return; /* get out of here! */
    }

    for (i=0; i<N-M && Y[i]<ymax ; i++); /* find the first node with max Y */
    F2_on = i;
    rsnc = 1; /* resonance! */


}


/**************************** update_R **********************************/

/* This population keeps track of the match between the output of F0
   (Qp) and the output of F1 (P). Resets are caused if the norm of R is
   less than the vigilance parameter. Note that ART2 vigilance is weird.
   Vigilance less than about 0.7 (with the default parameters) has no
   effect. See figure 7 and the discussion on Match-Reset Trade-Off (sec.
   IX, p. 4926) in the article. Keep in mind that here we are comparing
   Qp and P instead of the article's U and P.  Hence equations 22-25 will
   be a bit different. The program can easily be modified to base
   comparison on U instead of Qp, with similar results.  Because U is
   part of F1, it is more susceptible to Top-Down influence from F2, so
   that using U would be different from using Qp if the active F2
   category contains an element(s) that are not present in the input I */

double update_R(cntr)
     int cntr;
{
    int i;
    double ptmp=L2_norm(P,M), qtmp=L2_norm(Qp,M), utmp=L2_norm(U,M);
    double L2_R;
    int pf = (cntr%10==0 ? 1 : 0);

    /* Keep an eye out for spurious small norms. Only worrysome in very weird
       cases, if resets did not work well, or if no pattern came through. */
    if (qtmp < 0.1 || ptmp < 0.1 || utmp < 0.1) {
      fprintf(stderr,"%d Warning: }(Qp) = %5.4lf  }(p) = %5.4lf }(u)=%5.4lf\n",
	      cntr,qtmp,ptmp,utmp);
      fprintf(flog,"%d Warning: }(Qp) = %5.4lf  }(p) = %5.4lf }(u)=%5.4lf\n",
	      cntr,qtmp,ptmp,utmp);
  }

    /* compute R based on p-Q' comparison */
    for (i=0; i<M; i++)
      {
	  R[i]=(Qp[i]+c*P[i])/(c*ptmp+qtmp);
      }

    L2_R=L2_norm(R,M); /* Norm of R must be > rho (vigilance) */

    if (L2_R < rho) { /* Mismatch! Print blurb. */
	fprintf(stderr,"\n%d) L2_norm(R)=%4.3lf\n",cntr,L2_R);
	fprintf(flog,"\n%d) L2_norm(R)=%4.3lf\n",cntr,L2_R);
    }
    else
      if (cntr < 10) { /* No mismatch, only print at first */
	fprintf(stderr,"\n%d) L2_norm(R)=%4.3lf\n",cntr,L2_R);
	fprintf(flog,"\n%d) L2_norm(R)=%4.3lf\n",cntr,L2_R);
    }

    return(L2_R); /* return the norm of R */

}



/***************************** reset_F2 *******************************/
/* This function gets called if there has been an F2 reset. */

void reset_F2(p,cntr)
     int p,cntr;
{
    int j;

    fprintf(stderr,"%d) input %d caused a reset of node %d\n",cntr,p,F2_on);
    fprintf(flog,"%d) input %d caused a reset of node %d\n",cntr,p,F2_on);

    /* This is all the bookkeeping necessary if a mismatch occurred */
    Y[F2_on]=0.0; /* Kill activation */
    mis[F2_on]=-1; /* Record mismatch at this node */
    n_mis++; /* one more mismatch */
    rsnc = 0; /* No resonance */
    F2_on=-1; /* No active F2 node */
    IL[p]=-1; /* This pattern (p) is not coded by anyone! */

    /* Reset all F1 nodes. The big shortcoming of ART2, it needs to reset
       F1 whenever F2 is reset. ART3 gets around it very nicely. */
    for (j=0; j<M; j++)
      W[j]=P[j]=X[j]=Q[j]=V[j]=U[j]=0.0;
}

/***************************** resonate *******************************/
/* A stupid function that does not really need to be a function, but other
   things may be done here if useful to the user. */

void resonate(p,i)
     int p,i;
{
    fprintf(stderr,"** Pattern %d resonates with node %d\n",p,i);
    fprintf(flog,"** Pattern %d resonates with node %d\n",p,i);
}



/***************************** par_modify *******************************/
/* This funciton allows the user to vary any one of a number of system
   parameters. */

void par_modify()
{
    int which; /* which item to change */
    fprintf(flog,"Just called par_modify: ");

    do {
    /* This menu could be modified to be suited to whatever parameters are
       going to be accessed a lot. */
	fprintf(stderr,"1) vigilance (rho)\n");
	fprintf(stderr,"2) threshold (theta)\n");
	fprintf(stderr,"3) RK4 step size (h)\n");
	fprintf(stderr,"4) Number of integration steps (nsteps)\n");
	fprintf(stderr,"5) stability criterion (ac_dif)\n");
	fprintf(stderr,"6) LTM stability criterion (z_dif)\n");
	fprintf(stderr,"Which parameter ? ");
	scanf("%d",&which);

	switch (which) {
	  case 1 : fprintf(stderr,"Present value for rho is %6.3lf\nEnter new value --> ",rho);
	    fprintf(flog,"changed rho from %5.4lf",rho);
	    scanf("%lf",&rho);
	    fprintf(flog," to %5.4lf\n",rho);
	    break;
	  case 2 : fprintf(stderr,"Present value for theta is %6.3lf\nEnter new value --> ",theta);
	    fprintf(flog,"changed theta from %5.4lf",theta);
	    scanf("%lf",&theta);
	    fprintf(flog," to %5.4lf\n",theta);
	    break;
	  case 3 : fprintf(stderr,"Present value for h is %6.3lf\nEnter new value --> ",h);
	    fprintf(flog,"changed h from %5.4lf",h);
	    scanf("%lf",&h);
	    fprintf(flog," to %5.4lf\n",h);
	    break;
	  case 4 : fprintf(stderr,"Present value for nsteps is %d\nEnter new value --> ",nsteps);
	    fprintf(flog,"changed nsteps from %d",nsteps);
	    scanf("%d",&nsteps);
	    fprintf(flog," to %d\n",nsteps);
	    break;
	  case 5 : fprintf(stderr,"Present value for ac_dif is %6.3lf\nEnter new value --> ",ac_dif);
	    fprintf(flog,"changed ac_dif from %5.4lf",ac_dif);
	    scanf("%lf",&ac_dif);
	    fprintf(flog," to %5.4lf\n",ac_dif);
	    break;
	  case 6 : fprintf(stderr,"Present value for z_dif is %6.3lf\nEnter new value --> ",z_dif);
	    fprintf(flog,"changed z_dif from %5.4lf",z_dif);
	    scanf("%lf",&z_dif);
	    fprintf(flog," to %5.4lf\n",z_dif);
	    break;
	    default : which=0;
	    break;
	}
	fprintf(flog," to change case %d\n",which);
    } while (which != 0);
}



/***************************** show_setup *******************************/
/* This function gives a brief description of the present setup. Should
   be modified to suit the user's needs. Output goes to screen and to log
   file. */

void show_setup()
{
    int i;

    fprintf(stderr,"\nINPUT PATTERNS:\n"); /* only goes to screen */
    for (i=0; i<p_num; i++)
      p_display(i);

    if (ppc > 0) {
	fprintf(stderr,"\nSELECTED PATTERNS:\n");
	fprintf(flog,"\nSELECTED PATTERNS:\n");
	for (i=0; i<ppc; i++) {
	    fprintf(stderr,"No.%d in cycle is input pattern %d, coded at F2 node %d\n",i,c_p[i],IL[c_p[i]]);
	    fprintf(flog,"No.%d in cycle is input pattern %d, coded at F2 node %d\n",i,c_p[i],IL[c_p[i]]);
	}
    }

    fprintf(stderr,"\nPARAMETERS:\n");
    fprintf(stderr,"rho=%5.4lf  theta=%5.4lf  h=%4.3lf\nnsteps=%d ac_dif=%6.5lf z_dif=%5.4lf\n",rho,theta,h,nsteps,ac_dif,z_dif);
    fprintf(flog,"rho=%5.4lf  theta=%5.4lf  h=%4.3lf\nnsteps=%d ac_dif=%6.5lf z_dif=%5.4lf\n",rho,theta,h,nsteps,ac_dif,z_dif);

}



/***************************** init_weights *******************************/
/* This is a lower-level menu to deal with weights. Menu allows user to
   change, reset, display, or print weights. */

void init_weights()
{
    int i,j, choice=1;
    void BU_reset(), BU_set1(), TD_reset(), TD_set1(), show_weights(),
    print_weights(), zero_all();


    fprintf(stderr,"Initializing weights.\n");

    do {
	fprintf(stderr,"1) Initialize BU weights\n");
	fprintf(stderr,"2) Initialize TD weights\n");
	fprintf(stderr,"3) Initialize all weights\n");
	fprintf(stderr,"4) Manually set one BU weight\n");
	fprintf(stderr,"5) Manually set one TD weight\n");
	fprintf(stderr,"6) Show all weights\n");
	fprintf(stderr,"7) Print weights\n");
	fprintf(stderr,"8) Zero all weights\n");
	fprintf(stderr,"other) Return to main menu\n\n");
	fprintf(stderr,"---- Select a number --> ");
	scanf("%d",&choice);

	switch (choice) {
	  case 1 : BU_reset();
	    break;
	  case 2 : TD_reset();
	    break;
	  case 3 : BU_reset();
	    TD_reset();
	    break;
	  case 4 : BU_set1();
	    break;
	  case 5 : TD_set1();
	    break;
	  case 6 : show_weights();
	    break;
	  case 7 : print_weights();
	    break;
	  case 8: zero_all();
	    break;
	  default : choice = 0;
	}
    } while (choice != 0);

}


/***************************** BU_reset *******************************/
/* reset Bottom-Up weights. The value of the weights is such as to bias ART
   toward chosing a new category. See Sect. XI, p. 4927, of the article.
   Smaller, nonzero, random weights could also be chose here. */

void BU_reset()
{
    int i,j;

    fprintf(flog,"Just reset all TD weights.\n");

    for (i=0; i<M; i++)
      for (j=0; j<N-M; j++)
	Zij[i][j]=1.0/((1.0-d)*sqrt((double)M));
}


/***************************** BU_set1 *******************************/
/* Manually set an individual Bottom-Up weight. */

void BU_set1()
{
    int i,j;

    fprintf(stderr,"Enter i --> ");
    scanf("%d",&i);

    if (i < 0 || i >= M)
      {
	  fprintf(stderr,"Invalid weight index %d\n",i);
	  return;
      }

    fprintf(flog,"Resetting Zij[%d] to : ",i);
    for (j=0; j<N-M; j++)
      {
	  fprintf(stderr,"Enter Zij[%d][%d] (%4.3lf)--> ",i,j,Zij[i][j]);
	  scanf("%lf",&Zij[i][j]);
	  fprintf(flog,"%4.3lf ",Zij[i][j]);
      }
    fprintf(flog,"\n");
}


/***************************** TD_reset *******************************/
/* reset all Top-Down weights. Initial weights are set to zero, so as to
   satisfy the Match-Reset trade-off outlined in section IX. Small random
   numbers could also be chosen here. */

void TD_reset()
{
    int i,j;

    fprintf(flog,"Just reset all TD weights.\n");

    for (j=0; j<N-M; j++)
      for (i=0; i<M; i++)
	Zji[j][i]=0.0;
}

/***************************** TD_set1 *******************************/
/* Manually set an individual Top-Down weight. */

void TD_set1()
{
    int i,j;

    fprintf(stderr,"Enter j --> ");
    scanf("%d",&j);
    if (j < 0 || j >= N-M)
      {
	  fprintf(stderr,"Invalid weight index %d\n",j);
	  return;
      }

    fprintf(flog,"Setting Zji[%d] to: ",j);
    for (i=0; i<M; i++)
      {
	  fprintf(stderr,"Enter Zji[%d][%d] (%4.3lf)--> ",j,i,Zji[j][i]);
	  scanf("%lf",&Zji[j][i]);
	  fprintf(flog,"%4.3lf ",Zji[j][i]);
      }
    fprintf(flog,"\n");
}


/***************************** show_weights *******************************/
/* Print out weights on the screen and in the log file. */

void show_weights()
{
    int i,j;

    fprintf(stderr,"\nBU WEIGHTS:\n");
    fprintf(flog,"\nBU WEIGHTS:\n");
    for(i=0; i<M; i++)
      {
	  for(j=0; j<N-M; j++)
	    {
		fprintf(stderr,"%6.4lf ",Zij[i][j]);
		fprintf(flog,"%6.4lf ",Zij[i][j]);
	    }
	  fprintf(stderr,"\n");
	  fprintf(flog,"\n");
      }

    fprintf(stderr,"\nTD WEIGHTS:\n");
    fprintf(flog,"\nTD WEIGHTS:\n");
    for(j=0; j<N-M; j++)
      {
	  for(i=0; i<M; i++)
	    {
		fprintf(stderr,"%6.4lf ",Zji[j][i]);
		fprintf(flog,"%6.4lf ",Zji[j][i]);
	    }
	  fprintf(stderr,"\n");
	  fprintf(flog,"\n");
      }

}


/************************ zero_all ************************************/
/* This function resets all weights to 0.0. Useful for playing with the
   weights. */

void zero_all()
{
    int i,j;

    for (i=0; i<M; i++)
      for (j=0; j<N-M; j++)
	Zij[i][j]=Zji[j][i] = 0.0;
    fprintf(stderr,"Warning: ART2 will not work with all weights at zero.\n");
    fprintf(flog,"Reset all weights to zero!\n");

}


double zt1, zt2;

/************************* Zij equations *************************/
/* Bottom-Up learning law. I use temp variables here for speed. */

double dZij(i,z) /* Slow Learning */
     int i;
     double z;
{
    return(zt1*(zt2-z));
}

/************************* Zji equations *************************/
/* Top-Down learning law. I use temp variables here for speed. */

double dZji(i,z) /* Slow Learning */
     int i;
     double z;
{
    return(zt1*(zt2-z));
}


/************************* update_Zs ************************************/
/* This function updates all LTM traces using an RK4 solver. */

double update_Zs(cntr)
     int cntr;
{
    int i;
    int pf = (cntr%10==0);
    double *OZij=old1M, *OZji=old2M;
    /* Keep track of how much weights have changed for convergence check
       in update_F1 */
    double BUdiff=0.0;
    double TDdiff;

    for (i=0; i<M; i++)
      {
	  OZij[i]=Zij[i][F2_on];
	  OZji[i]=Zji[F2_on][i];
/* See discussion in the main header comment. If the values: zt1=d and
   zt2=P[i] are used, then you get the "dips" I talk about. If you
   don't like that, use the following two lines instead. */
	  zt1=d*(1.0-d);
	  zt2=U[i]/(1.0-d);
/*	  zt1=d;
	  zt2=P[i]; */
	  Zij[i][F2_on]=RungeKutta(dZij,i,Zij[i][F2_on],h);
	  Zji[F2_on][i]=RungeKutta(dZji,i,Zji[F2_on][i],h);
	  BUdiff += fabs(OZij[i]-Zij[i][F2_on]);
      }

    TDdiff=vec_diff(Zji[F2_on],OZji,M);

    if (pf) {

	fprintf(stderr,"%d) (BU,TD) vector LTM changes are (%5.4lf,%5.4lf)\n",
		cntr,BUdiff,TDdiff);

	fprintf(flog,"%d) (BU,TD) vector LTM changes are (%5.4lf,%5.4lf)\n",
		cntr,BUdiff,TDdiff);
    }

    return(BUdiff+TDdiff); /* convergence check for update_F1() */
}


/**************************** print_weights ******************************/
/* This function dumps the weights to the .dat file. */

void print_weights()
{
    int i,j;

    fprintf(fout,"\t\t Bottom-up weights:\n\n");
    for (i=0; i<M; i++) {
	for (j=0; j<N-M; j++)
	  fprintf(fout,"%5.4lf ",Zij[i][j]);
	fprintf(fout,"\n");
    }

    fprintf(fout,"\n\t\t Top-down weights:\n\n");
    for (j=0; j<N-M; j++) {
	for (i=0; i<M; i++)
	  fprintf(fout,"%5.4lf ",Zji[j][i]);
	fprintf(fout,"\n");
    }
    fprintf(fout,"\n");
}



/**************************** alloc_pops ****************************/
/* this function allocates space for all the populations based on the values
   M and N in the input file. */

void alloc_pops() {
    int i;

    X=(double *)calloc(M,sizeof(double));
    Xp=(double *)calloc(M,sizeof(double));
    W=(double *)calloc(M,sizeof(double));
    Wp=(double *)calloc(M,sizeof(double));
    U=(double *)calloc(M,sizeof(double));
    Up=(double *)calloc(M,sizeof(double));
    P=(double *)calloc(M,sizeof(double));
    Pp=(double *)calloc(M,sizeof(double));
    V=(double *)calloc(M,sizeof(double));
    Vp=(double *)calloc(M,sizeof(double));
    Q=(double *)calloc(M,sizeof(double));
    Qp=(double *)calloc(M,sizeof(double));
    R=(double *)calloc(M,sizeof(double));
    Y=(double *)calloc(N-M,sizeof(double));
    mis=(double *)calloc(N-M,sizeof(double));

    Zij=(double **) calloc(M,sizeof(double *));
    for (i=0; i<M; i++)
      Zij[i]=(double *) calloc(N-M,sizeof(double));

    Zji=(double **) calloc(N-M,sizeof(double *));
    for (i=0; i<N-M; i++)
      Zji[i]=(double *) calloc(M,sizeof(double));

    /* Some necessary temp storage */
    old1M=(double *) calloc(M,sizeof(double));
    old2M=(double *) calloc(M,sizeof(double));

}



/**************************** free_pops ****************************/
/* this function frees all allocated space for populations. */

void free_pops() {
    int i;

    free(X);
    free(Xp);
    free(W);
    free(Wp);
    free(U);
    free(Up);
    free(P);
    free(Pp);
    free(V);
    free(Vp);
    free(Q);
    free(Qp);
    free(R);
    free(Y);
    free(mis);

    for (i=0; i<M; i++)
      free(Zij[i]);
    free(Zij);

    for (i=0; i<N-M; i++)
      free(Zji[i]);
    free(Zji);

    free(old1M);
    free(old2M);
}
