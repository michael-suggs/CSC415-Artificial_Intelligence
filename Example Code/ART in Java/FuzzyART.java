package kcdc.nn.art;

import kcdc.nn.*;

public class FuzzyART extends ART
{
  public FuzzyART() { }

  public FuzzyART(int num_inputs, int input_style)
  { super(num_inputs, input_style); }

  /** Initialize FuzzyART network. */
  public void init()
  {
    super.init();
    setInputType(Pattern.ANALOG);
  }

  public String toString()
  { return new String("FuzzyART(" + neurons +")"); }

  protected double intersect(double x, double y)
  { return Fuzzy.intersect(x, y);  }

  public void addWinner(Pattern pattern)
  {
    // Add extra winning neuron
    winner = (ARTNeuron) new FuzzyARTNeuron(pattern.getNumInputs());
    winner_index = neurons.size();
    neurons.addElement(winner);
    winner.setID(winner_index);
    winner.activate(pattern);
    categorize(winner);
    winner.learn(pattern, 1);
    resizes++;
    done = false;
    size = neurons.size();
    multicaster.artResize(new ARTEvent(this, ARTEvent.ART_RESIZE));
  }  
}
