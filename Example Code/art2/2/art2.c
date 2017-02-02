#include "stdio.h"
#include "math.h"
#define M 10
#define N 8
#define STREQ(s1,s2) (strcmp((s1),(s2)) == 0)

/*****************************************************************/
/**	art2 algorithm						**/
/** simulates the art2 algorithm for clustering vectors		**/
/**	Steve Lehar Sept 89					**/
/*****************************************************************/

/**** global variables & functions ****/
int reset;
float w[M],p[M],oldp[M],x[M],q[M],v[M],u[M],r[M],topdown[M];
float w1[M],p1[M],x1[M],q1[M],v1[M],u1[M];
float rho,normp,normu,normv,normw,normr;
float normq1,normp1,normu1,normv1,normw1,normr1;
float zup[M][N],zdn[N][M],oldzdn[M];
float I[M], F1[M], F2[N],F2val[N];
int F2winner,F2reset[N];
float pchange=1.0,zchange=1.0,diff();
float a=5.0, b=5.0, c=0.225, e=0.000001, theta = 0.3,del_t = 0.1;
char pattern[80];
FILE *fp;
int iter,learning,nreset=0;
float f(),floatval();

main(argc,argv)
int argc;
char *argv[];
{
  int i,j,k;
  char key;

  /**** initialize ****/
  init_weights();
  get_commandline_args(argc,argv);

  /**** learn training patterns ****/
  for(i=0;i<10;i++){
    fscanf(fp,"%s ",pattern);
    learning = 1;
/****
    if(STREQ(pattern,"{") || STREQ(pattern,"F")) learning = 0;
    else learning = 1;
****/
    read_input();
    init_reset();
    print_input();
  restart:;
    init_nodes();

    /**** equilibrate f0 ****/
    for(iter=0;iter<10;iter++){
      update_f0();
    }

    /**** equilibrate f1 ****/
    for(iter=0;iter<10;iter++){
      update_f1();
    }

    /**** equilibrate f1 & f2 ****/
    for(iter=0;iter<10;iter++){
      update_f1();
      update_f2();
      update_r();
    }

    /**** check for reset ****/
    if(reset){
      printf("RESET\n");
      reset = 0;
      nreset++;
      if(nreset==N)goto nextpattern;
      if(learning) goto restart;
    }

    if(!learning) goto skip;

    /**** learn resonant pattern ****/
    for(k=0;k<1000;k++){
      update_zup();
      update_zdn();
      update_f1();
    }


skip:;
    print_vals();
    if(is_arg("-v",argc,argv)) print_f0();
    print_f1();
    print_f2();
    print_z(F2winner);

}
    printf("\n");
nextpattern:;

  fclose(fp);
}


/****
    zchange = diff(zdn[F2winner],oldzdn,N);
    pchange = diff(p,oldp,M);
    print_f2();
    print_all();
****/

/*****************************************************************/
/**	function f()						**/
/** returns 0 if x<theta, else returns theta			**/
/*****************************************************************/
float f(x)
float x;
{
  if(x<theta) return(0.0); else return(x);
}

/*****************************************************************/
/**	diff							**/
/*****************************************************************/
float diff(v1,v2,n)
float v1[],v2[];
int n;
{
  int i;
  float d;

  d = 0.0;
  for(i=0;i<n;i++){
    d += (v1[i] - v2[i])*(v1[i] - v2[i]);
    v2[i] = v1[i];
  }
  return(d);
}

/*****************************************************************/
/**	initialize weights					**/
/*****************************************************************/
init_weights()
{
  int i,j;

  /**** initialize weights ****/
  for(j=0;j<N;j++){
    for(i=0;i<M;i++){
      zup[i][j] = 1.0/(0.2*sqrt((float)M));
      zdn[j][i] = 0.0;
    }
  }

}

/*****************************************************************/
/**	init_reset						**/
/*****************************************************************/
init_reset()
{
  int j;

  for(j=0;j<N;j++){
    F2reset[j] = 0;
  }
  nreset = 0;
}

/*****************************************************************/
/**	initialize nodes					**/
/*****************************************************************/
init_nodes()
{
  int i,j;

  /**** initialize nodes ****/
  for(i=0;i<M;i++){
    F1[i] = 0.0;
    w1[i] = 0.0;
    x1[i] = 0.0;
    v1[i] = 0.0;
    u1[i] = 0.0;
    p1[i] = 0.0;
    q1[i] = 0.0;
    w[i] = 0.0;
    x[i] = 0.0;
    v[i] = 0.0;
    u[i] = 0.0;
    p[i] = 0.0;
    q[i] = 0.0;
  }
  for(j=0;j<M;j++){
    F2[j] = 0.0;
  }

}

/*****************************************************************/
/**	get_commandline_args					**/
/*****************************************************************/
get_commandline_args(argc,argv)
int argc;
char *argv[];
{

  /**** open data file & get rho ****/
  if (!is_arg("-i",argc,argv)){
    printf("ERROR no input file given [-i file]\n");
    exit(1);
  }
  if (!is_arg("-r",argc,argv)){
    printf("ERROR no rho value given [-r rho]\n");
    exit(1);
  }
  fp = fopen(stringval("-i",argc,argv),"r");
  rho = floatval("-r",argc,argv);
}

/*****************************************************************/
/**	read_input						**/
/*****************************************************************/
read_input()
{
  int i,j;

  /**** read input vector ****/
  for(i=0;i<M;i++){
    fscanf(fp,"%f",&I[i]);
  }
}

/*****************************************************************/
/**	update_f0						**/
/*****************************************************************/
update_f0()
{
    int i,j;

    /**** update w nodes ****/
    normw1 = 0.0;
    for(i=0;i<M;i++){
      w1[i] = I[i]  + a*u1[i];
      normw1 += w1[i] * w1[i];
    }
    normw1 = sqrt(normw1);

    /**** update p nodes ****/
    normp1 = 0.0;
    for(i=0;i<M;i++){
      p1[i] = u1[i];
      normp1 += p1[i] * p1[i];
    }
    normp1 = sqrt(normp1);

    /**** update x nodes ****/
    for(i=0;i<M;i++){
      x1[i] = w1[i] / (e+normw1);
    }

    /**** update q nodes ****/
    normq1 = 0.0;
    for(i=0;i<M;i++){
      q1[i] = p1[i] / (e+normp1);
      normq1 += q1[i] * q1[i];
    }
    normq1 = sqrt(normq1);

    /**** update v nodes ****/
    normv1 = 0.0;
    for(i=0;i<M;i++){
      v1[i] = f(x1[i]) + b*f(q1[i]);
      normv1 += v1[i] * v1[i];
    }
    normv1 = sqrt(normv1);

    /**** update u nodes ****/
    for(i=0;i<M;i++){
      u1[i] = v1[i] / (e+normv1);
    }

}

/*****************************************************************/
/**	update_f1						**/
/*****************************************************************/
update_f1()
{
    int i,j;

    /**** update w nodes ****/
    normw = 0.0;
    for(i=0;i<M;i++){
      w[i] = q1[i]  + a*u[i];
      normw += w[i] * w[i];
    }
    normw = sqrt(normw);

    /**** update top-down value ****/
    for(i=0;i<M;i++){
      topdown[i] = 0.0;
      for(j=0;j<N;j++){
        topdown[i] += F2[j] * zdn[j][i];
      }
    }

    /**** update p nodes ****/
    normp = 0.0;
    for(i=0;i<M;i++){
      p[i] = u[i] + topdown[i];
      normp += p[i] * p[i];
    }
    normp = sqrt(normp);

    /**** update x nodes ****/
    for(i=0;i<M;i++){
      x[i] = w[i] / (e+normw);
    }

    /**** update q nodes ****/
    for(i=0;i<M;i++){
      q[i] = p[i] / (e+normp);
    }

    /**** update v nodes ****/
    normv = 0.0;
    for(i=0;i<M;i++){
      v[i] = f(x[i]) + b*f(q[i]);
      normv += v[i] * v[i];
    }
    normv = sqrt(normv);

    /**** update u nodes ****/
    normu = 0.0;
    for(i=0;i<M;i++){
      u[i] = v[i] / (e+normv);
      normu += u[i] * u[i];
    }
    normu = sqrt(normu);

}

/*****************************************************************/
/**	update_f2						**/
/*****************************************************************/
update_f2()
{
   int i,j;
   float max;

  /**** compute F2 ****/
  for(j=0;j<N;j++){
    F2[j] = 0.0;
    for(i=0;i<M;i++){
      F2[j]+= p[i] * zup[i][j];
    }
    F2val[j] = F2[j];
  }

  /**** compute F2winner ****/
  max = 0.0;
  for(j=0;j<N;j++){
    if(F2[j] > max && !F2reset[j]){
      max = F2[j];
      F2winner = j;
    }
    F2[j] = 0.0;
  }
  F2[F2winner] = 0.8;
}

/*****************************************************************/
/**	update_r						**/
/*****************************************************************/
update_r()
{
  int i,j;
  float cnormp;

  /**** compute cnormp ****/
  cnormp = 0.0;
  for(i=0;i<M;i++){
    cnormp += c * p[i] * p[i];
  }
  cnormp = sqrt(cnormp);


  /**** compute reset ****/
  normr = 0.0;
  for(i=0;i<M;i++){
    r[i] = (q1[i] + c*p[i]) / (e + normq1 + c*normp);
    normr += r[i] * r[i];
  }
  normr = sqrt(normr);

  if(rho/(e+normr) > 1.0) reset = 1; else reset = 0;
  if(reset){
    F2reset[F2winner] = 1.0;
  }

}
/*****************************************************************/
/**	update_zup						**/
/*****************************************************************/
update_zup()
{
  int i,j;

  for(i=0;i<M;i++){
    zup[i][F2winner] += del_t * (p[i] - zup[i][F2winner]);
  }

}

/*****************************************************************/
/**	update_zdn						**/
/*****************************************************************/
update_zdn()
{
   int i,j;

  /**** set top-down weights ****/
  for(i=0;i<M;i++){
    zdn[F2winner][i] += del_t * (p[i] - zdn[F2winner][i]);
  }

}

/*****************************************************************/
/** 	print_f0						**/
/*****************************************************************/
print_f0()
{
  int i,j;

  /**** print values ****/
  printf("**** F 0  iteration %2d ****\n",iter);
  for(i=0;i<M;i++){
    printf("%s%2d: I=%5.2f  w1=%5.2f  x1=%5.2f  v1=%5.2f  ",
                   pattern, i, I[i], w1[i], x1[i], v1[i]);
    printf("u1=%5.2f  p1=%5.2f  q1=%5.2f\n",
            u1[i], p1[i], q1[i]);
  }
}

/*****************************************************************/
/**	print_f1						**/
/*****************************************************************/
print_f1()
{
  int i,j;

  /**** print F1 values ****/
  printf("w:   ");
  for(i=0;i<M;i++){
    printf("%5.2f ",w[i]);
  }
  printf("\n");
  printf("x:   ");
  for(i=0;i<M;i++){
    printf("%5.2f ",x[i]);
  }
  printf("\n");
  printf("v:   ");
  for(i=0;i<M;i++){
    printf("%5.2f ",v[i]);
  }
  printf("\n");
  printf("u:   ");
  for(i=0;i<M;i++){
    printf("%5.2f ",u[i]);
  }
  printf("\n");
  printf("p:   ");
  for(i=0;i<M;i++){
    printf("%5.2f ",p[i]);
  }
  printf("\n");
  printf("q:   ");
  for(i=0;i<M;i++){
    printf("%5.2f ",q[i]);
  }
  printf("\n");
}

/*****************************************************************/
/**	print_f2						**/
/*****************************************************************/
print_f2()
{
  int i,j;

  /**** print f2 values ****/
  printf("F2:  ");
  for(j=0;j<N;j++){
    printf("%5.2f ",F2val[j]);
  }
  printf("\n");

  /**** underline winner ****/
  printf("     ");
  for(j=0;j<N;j++){
    if(j==F2winner) printf(";WINS;");
    else printf("      ");
  }
  printf("\n");
}

/*****************************************************************/
/**	print_z							**/
/*****************************************************************/
print_z(node)
int node;
{
  int i,j;

  printf("zup%1d:",node);
  for(i=0;i<M;i++){
    printf("%5.2f ",zup[i][node]);
  }
  printf("\n");
  printf("zdn%1d:",node);
  for(i=0;i<M;i++){
    printf("%5.2f ",zdn[node][i]);
  }
  printf("\n");
}

/*****************************************************************/
/**	print_input						**/
/*****************************************************************/
print_input()
{
  int i,j;

  /**** print input ****/
  printf("\n %s:  ",pattern);
  for(i=0;i<M;i++){
    printf("%5.2f ",I[i]);
  }
  printf("\n");
}

/*****************************************************************/
/** 	print_vals						**/
/*****************************************************************/
print_vals()
{
  printf("rho=%5.2f normr=%5.2f\twinner=%1d",rho,normr,F2winner);
  if(reset)printf(" RESET");
  if(learning) printf(" learning");
  printf("\n");
}

/*****************************************************************/
/** 	print_all						**/
/*****************************************************************/
print_all(iter)
int iter;
{
  int i,j;

  /**** print values ****/
  printf("**** iteration %2d ****\n",iter);
  for(i=0;i<3;i++){
    printf("%s I=%5.2f  v1=%5.2f  q1=%5.2f  v=%5.2f  ",
                   pattern,I[i], v1[i], q1[i], v[i]);
    printf("p=%5.2f  r=%5.2f \n",
            p[i], r[i]);
  }
  printf("normr=%5.2f ",normr);
  if(reset) printf("RESET\n");
  printf("\n");

}
