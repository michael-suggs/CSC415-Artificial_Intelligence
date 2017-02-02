/*---------------------------- RungeKutta --------------------*/


double RungeKutta(func,i,x,h)
     double (*func)(), x, h;
     int i;
{
  double delta1, delta2, delta3, delta4;

  delta1 = h * (*func)(i,x);
  delta2 = h * (*func)(i,x + delta1/2.0);
  delta3 = h * (*func)(i,x + delta2/2.0);
  delta4 = h * (*func)(i,x + delta3);

  return(x+(delta1+2.0*delta2+2.0*delta3+delta4)/6.0);
}


