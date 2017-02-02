package kcdc.nn.art;

import java.util.*;

import kcdc.nn.*;

public class FuzzyARTNeuron extends ARTNeuron
{
  public FuzzyARTNeuron()
  { }
  
  public FuzzyARTNeuron(int n)
  { super(n); }

  public String toString()
  { return new String("FuzzyARTNeuron("+input+", "+output+", "+ weights+")"); }

  /** Fuzzy set intersection between two numbers [0,1] */
  protected double intersect(double x, double y)
  { return Fuzzy.intersect(x, y); }
}

