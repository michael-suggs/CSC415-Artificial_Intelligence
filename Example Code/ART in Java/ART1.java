package kcdc.nn.art;

import kcdc.nn.*;

public class ART1 extends ART
{
  public ART1() { }

  public ART1(int num_inputs, int input_style)
  { super(num_inputs, input_style); }

  /** Initialize ART1 network. */
  public void init()
  {
    super.init();
    setInputType(Pattern.BINARY);
  }

  public String toString()
  { return new String("ART1(" + neurons +")"); }

  /** Normal intersection of two numbers [0,1] */
  protected double intersect(double x, double y)
  { return x * y; }

  public void addWinner(Pattern pattern)
  {
    // Add extra winning neuron
    winner = (ARTNeuron) new ART1Neuron(pattern.getNumInputs());
    neurons.addElement(winner);
    winner_index = neurons.size() - 1;
    winner.activate(pattern);
    categorize(winner);
    winner.learn(pattern, 1);

//     System.err.println("Added winner " + winner +
// 		       " to neurons " + neurons +
// 		       " at index " + winner_index);

    resizes++;
    done = false;
    size = neurons.size();
    multicaster.artResize(new ARTEvent(this, ARTEvent.ART_RESIZE));
  }  
}
