package kcdc.nn.art;

import java.util.*;

import kcdc.nn.*;

public class MapNeuron
{
  public int id = 0;
  public int num_inputs = 0;
  public Vector weights = new Vector();
  public double sum_weights = 0.0;

  public double input = 0.0;
  public double output = 0.0;

  public boolean activated = false;
  public boolean eligible = true;

  /** A small number to avoid div by zero errors? */
  static final public double ALPHA = 0.1;

  public MapNeuron()
  {
    id = 0;
    weights = new Vector();
    sum_weights = 0.0;
    input = 0.0;
    output = 0.0;
    eligible = true;
  }
  
  public MapNeuron(int n)
  {
    id = 0;
    weights = new Vector();
    sum_weights = 0.0;
    input = 0.0;
    output = 0.0;
    eligible = true;
    for (int i = 0; i < n; i++)
      addWeight();
  }

  public String toString()
  { return new String("MapNeuron("+input+", "+output+", "+ weights+")"); }

  public void activate(Pattern pattern)
  {
    input = 0.0;
    output = 0.0;

    int winner = ((ART)pattern.getSource()).getWinnerIndex();
    if (winner != -1)
    {
      // Calculate new input
      input = getWeight(winner);
      // Calculate new output
      output = input;
    }

    activated = true;
  }

  public void learn(Pattern pattern, double rate)
  {
    if (!activated)
      return; // NNException

    for (int i = 0; i < pattern.getNumInputs(); i++)
      if (rate == 1.0) // Fast learning
	setWeight(i, intersect(pattern.getInput(i), getWeight(i)));
      else // Slow learning
	setWeight(i, ((rate * intersect(pattern.getInput(i), getWeight(i))) +
		      ((1.0 - rate) * getWeight(i))));

    activated = false;
  }
  
  public double intersect(double x, double y)
  { return x * y; }
  
  public int getID()
  { return id; }

  public boolean isActivated()
  { return activated; }

  public boolean isEligible()
  { return eligible; }

  public int getNumInputs()
  { return num_inputs; }

  public double[] getWeights()
  {
    double[] w = new double[weights.size()]; 
    for (int i = 0; i < weights.size(); i++)
      w[i] = getWeight(i);
    return w;
  }

  public double getWeight(int i)
  { return ((Double)weights.elementAt(i)).doubleValue(); }

  public void addWeight()
  {
    weights.addElement(new Double(1.0));
    sum_weights += 1.0;
    num_inputs++;
  }

  public void addWeight(double w)
  {
    weights.addElement(new Double(w));
    sum_weights += w;
    num_inputs++;
  }

  protected void setWeight(int i, double w)
  {
    sum_weights += w - getWeight(i);
    weights.setElementAt(new Double(w), i);
  }

  public double getSumWeights()
  { return sum_weights; }

  public double getInput()
  { return input; }

  public double getOutput()
  { return output; }
}

