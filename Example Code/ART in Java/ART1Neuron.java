package kcdc.nn.art;

import java.util.*;

import kcdc.nn.*;

public class ART1Neuron extends ARTNeuron
{
  public ART1Neuron()
  { }
  
  public ART1Neuron(int n)
  { super(n); }

  public String toString()
  { return new String("ART1Neuron("+input+", "+output+", "+ weights+")"); }

  public double intersect(double x, double y)
  { return x * y; }
}

