package kcdc.nn.art;

import kcdc.nn.*;

public class Fuzzy
{
  /** Fuzzy set intersection between two numbers [0,1] */
  public static final double intersect(double x, double y)
  {
    if (x > y) return y;
    else return x;
  }

  /** Fuzzy set intersection between two arrays of numbers [0,1] */
  public static final double intersect(double[] x, double[] y)
  {
    double intersection = 0.0;
    for (int i = 0; i < x.length; i++)
      intersection += Fuzzy.intersect(x[i], y[i]);
    return intersection;
  }

  /** Fuzzy set intersection between two Patterns */
  public static final double intersect(Pattern x, Pattern y)
  {
    double intersection = 0.0;
    for (int i = 0; i < x.getSize(); i++)
      intersection += Fuzzy.intersect(x.getInput(i), y.getInput(i));
    return intersection;
  }

  /** Fuzzy set union between two numbers [0,1] */
  public static final double union(double x, double y)
  {
    if (x > y) return x;
    else return y;
  }

  /** Fuzzy set union between two arrays of numbers [0,1] */
  public static final double union(double[] x, double[] y)
  {
    double union = 0.0;
    for (int i = 0; i < x.length; i++)
      union += Fuzzy.union(x[i], y[i]);
    return union;
  }

  /** Fuzzy set union between two Patterns */
  public static final double union(Pattern x, Pattern y)
  {
    double union = 0.0;
    for (int i = 0; i < x.getSize(); i++)
      union += Fuzzy.union(x.getInput(i), y.getInput(i));
    return union;
  }
}
