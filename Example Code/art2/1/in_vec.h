/***************************** in_vec.h *******************************/

#include <stdio.h>
#include "std_fun.h"

extern FILE *datain, *fopen();
extern int p_dim, p_num, *IL;
extern double **I, *p_mag;
extern int ppc, *c_p;

extern void get_inputs();
extern void select_one();
extern void select_cycle();
extern void init_all();
extern void new_learning();
extern void free_all();
extern void p_normalize();
extern void p_display();
